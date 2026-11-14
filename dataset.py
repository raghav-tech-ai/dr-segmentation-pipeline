"""
dataset.py -- PyTorch Dataset for the prepared IDRiD segmentation data.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

LESION_TYPES = ["MA", "HE", "EX", "SE", "OD"]


class IDRiDSegDataset(Dataset):
    def __init__(self, root, split, augment=False):
        self.img_dir = Path(root) / "images" / split
        self.mask_dir = Path(root) / "masks" / split
        self.names = sorted(p.stem for p in self.img_dir.glob("*.png"))
        if not self.names:
            raise FileNotFoundError(f"No images found in {self.img_dir}. Run prepare_dataset.py first.")
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_bgr = cv2.imread(str(self.img_dir / f"{name}.png"))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        masks = []
        for lesion in LESION_TYPES:
            m = cv2.imread(str(self.mask_dir / f"{name}_{lesion}.png"), cv2.IMREAD_GRAYSCALE)
            masks.append((m > 0).astype(np.float32))
        mask = np.stack(masks, axis=0)  # 5 x H x W

        if self.augment:
            img, mask = self._augment(img, mask)

        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy())  # 3 x H x W
        mask_t = torch.from_numpy(mask.copy())
        return img_t, mask_t

    @staticmethod
    def _augment(img, mask):
        # mask is 5 x H x W, img is H x W x 3 -- keep spatial transforms in sync
        if np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, :, ::-1])
        if np.random.rand() < 0.5:
            img = np.ascontiguousarray(img[::-1, :, :])
            mask = np.ascontiguousarray(mask[:, ::-1, :])
        k = np.random.choice([0, 1, 2, 3])
        if k:
            img = np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
            mask = np.ascontiguousarray(np.rot90(mask, k, axes=(1, 2)))
        if np.random.rand() < 0.3:
            factor = np.random.uniform(0.85, 1.15)
            img = np.clip(img * factor, 0, 1)
        return img, mask
