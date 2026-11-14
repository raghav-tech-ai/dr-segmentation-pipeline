"""
losses.py -- Combined BCE + Dice loss and per-class Dice/IoU metrics.

Lesion masks are extremely imbalanced (MA pixels are <<1% of a patch),
so plain BCE alone will collapse to predicting all-background. Dice
loss directly penalizes that.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LESION_TYPES = ["MA", "HE", "EX", "SE", "OD"]


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], probs.shape[1], -1)
    targets = targets.reshape(targets.shape[0], targets.shape[1], -1)

    intersection = (probs * targets).sum(dim=2)
    union = probs.sum(dim=2) + targets.sum(dim=2)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        d_loss = dice_loss(logits, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * d_loss


@torch.no_grad()
def per_class_dice(logits, targets, threshold=0.5, eps=1e-6):
    """Returns a dict {lesion_name: dice_score} averaged over the batch."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    scores = {}
    for i, name in enumerate(LESION_TYPES):
        p = preds[:, i].reshape(preds.shape[0], -1)
        t = targets[:, i].reshape(targets.shape[0], -1)
        intersection = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        # if a sample has no GT and no prediction for this class, dice = 1
        dice = torch.where(union > 0, (2 * intersection + eps) / (union + eps), torch.ones_like(union))
        scores[name] = dice.mean().item()
    return scores
