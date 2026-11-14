"""
predict.py -- Run the trained UNet on a single fundus image and save a
lesion overlay visualization. Uses the same enhancement.py pipeline as
training, in 'full' resize mode, so preprocessing matches train time.

Usage:
    python predict.py --image path/to/fundus.jpg --checkpoint checkpoints/best.pt --out result.png
"""

import argparse

import cv2
import numpy as np
import torch

from enhancement import enhance_image
from model import UNet

LESION_TYPES = ["MA", "HE", "EX", "SE", "OD"]
LESION_COLORS = {  # BGR for cv2 drawing
    "MA": (0, 0, 255),     # red
    "HE": (0, 128, 255),   # orange
    "EX": (0, 255, 255),   # yellow
    "SE": (255, 255, 0),   # cyan
    "OD": (255, 0, 255),   # magenta
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="prediction_overlay.png")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--base_features", type=int, default=32,
                     help="Must match the value used in train.py")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    enhanced, fov, _ = enhance_image(img_rgb, target_size=(args.size, args.size))

    model = UNet(in_channels=3, out_channels=5, base_features=args.base_features).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.from_numpy(enhanced.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()  # 5 x H x W

    overlay = (enhanced * 255).astype(np.uint8).copy()
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    for i, lesion in enumerate(LESION_TYPES):
        mask = (probs[i] > args.threshold).astype(np.uint8)
        color = np.array(LESION_COLORS[lesion], dtype=np.uint8)
        colored = np.zeros_like(overlay_bgr)
        colored[mask == 1] = color
        overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, colored, 0.5, 0)
        pixel_count = int(mask.sum())
        print(f"{lesion}: {pixel_count} px flagged (prob>{args.threshold})")

    cv2.imwrite(args.out, overlay_bgr)
    print(f"Saved overlay to {args.out}")


if __name__ == "__main__":
    main()
