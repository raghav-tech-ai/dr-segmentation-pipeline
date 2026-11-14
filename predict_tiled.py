"""
predict_tiled.py -- Correct inference for the patch-trained UNet.

IMPORTANT: predict.py resizes the whole input image to (size x size) in
one shot before running the model. That does NOT match how this model
was trained -- training patches were 256x256 crops taken from the
FULL-RESOLUTION original image (no shrinking), specifically so tiny
lesions (microaneurysms) wouldn't get compressed away. Feeding a
whole-image resize at inference time re-introduces exactly the problem
patch-based training was meant to solve.

This script instead:
  1. Enhances the new image at full native resolution (no resize)
  2. Slides a (patch_size x patch_size) window across it with overlap
  3. Runs the model on each window
  4. Averages overlapping predictions back into one full-resolution
     probability map per lesion class
  5. Thresholds and draws the color overlay

Use --patch_size / --stride matching what you used in prepare_dataset.py
(default 256 / stride 192 gives good overlap-averaging quality).

Usage:
    python predict_tiled.py --image path/to/fundus.jpg \
        --checkpoint checkpoints/best.pt --base_features 32 \
        --patch_size 256 --stride 192 --out result.png
"""

import argparse
import os

import cv2
import numpy as np
import torch

from enhancement import enhance_image
from model import UNet

LESION_TYPES = ["MA", "HE", "EX", "SE", "OD"]
LESION_COLORS = {  # BGR for cv2 drawing
    "MA": (0, 0, 255),
    "HE": (0, 128, 255),
    "EX": (0, 255, 255),
    "SE": (255, 255, 0),
    "OD": (255, 0, 255),
}
LESION_LABELS = {
    "MA": "Microaneurysms",
    "HE": "Haemorrhages",
    "EX": "Hard Exudates",
    "SE": "Soft Exudates",
    "OD": "Optic Disc (landmark)",
}


def draw_legend(overlay_bgr):
    """Burns a color-coded legend box into the bottom-left corner of the image."""
    H, W = overlay_bgr.shape[:2]

    # Scale everything relative to image width so it looks right on any resolution
    scale = W / 1200.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55 * scale
    thickness = max(1, int(round(1 * scale)))
    row_h = int(34 * scale)
    swatch_size = int(20 * scale)
    pad = int(14 * scale)

    n_rows = len(LESION_TYPES)
    box_h = pad * 2 + row_h * n_rows
    box_w = int(360 * scale)
    x0, y0 = pad, H - box_h - pad

    # semi-transparent dark background so text is readable over any part of the photo
    overlay_copy = overlay_bgr.copy()
    box_only = overlay_bgr.copy()
    cv2.rectangle(box_only, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    blended = cv2.addWeighted(box_only, 0.55, overlay_copy, 0.45, 0)
    overlay_bgr[:, :] = blended

    for i, lesion in enumerate(LESION_TYPES):
        row_y = y0 + pad + i * row_h
        swatch_y = row_y + (row_h - swatch_size) // 2
        color = LESION_COLORS[lesion]
        cv2.rectangle(overlay_bgr, (x0 + pad, swatch_y),
                       (x0 + pad + swatch_size, swatch_y + swatch_size), color, -1)
        cv2.rectangle(overlay_bgr, (x0 + pad, swatch_y),
                       (x0 + pad + swatch_size, swatch_y + swatch_size), (255, 255, 255), 1)
        label = f"{lesion} - {LESION_LABELS[lesion]}"
        text_x = x0 + pad + swatch_size + int(10 * scale)
        text_y = row_y + row_h // 2 + int(6 * scale)
        cv2.putText(overlay_bgr, label, (text_x, text_y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    return overlay_bgr


def sliding_window_predict(model, enhanced, fov, patch_size, stride, device, batch_size=8):
    H, W = enhanced.shape[:2]
    prob_sum = np.zeros((5, H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    y_steps = list(range(0, max(H - patch_size, 0) + 1, stride))
    x_steps = list(range(0, max(W - patch_size, 0) + 1, stride))
    if not y_steps:
        y_steps = [0]
    if not x_steps:
        x_steps = [0]
    # make sure the bottom/right edges are covered even if step doesn't land exactly
    if y_steps[-1] + patch_size < H:
        y_steps.append(H - patch_size)
    if x_steps[-1] + patch_size < W:
        x_steps.append(W - patch_size)

    coords = [(y, x) for y in y_steps for x in x_steps
              if 0 <= y <= H - patch_size and 0 <= x <= W - patch_size]

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        batch_imgs = []
        for (y, x) in batch_coords:
            patch = enhanced[y:y + patch_size, x:x + patch_size]
            batch_imgs.append(patch.transpose(2, 0, 1))
        batch_t = torch.from_numpy(np.stack(batch_imgs)).float().to(device)

        with torch.no_grad():
            logits = model(batch_t)
            probs = torch.sigmoid(logits).cpu().numpy()  # B x 5 x patch x patch

        for (y, x), p in zip(batch_coords, probs):
            prob_sum[:, y:y + patch_size, x:x + patch_size] += p
            weight_sum[y:y + patch_size, x:x + patch_size] += 1.0

    weight_sum = np.clip(weight_sum, 1e-6, None)
    avg_probs = prob_sum / weight_sum[None, :, :]
    return avg_probs  # 5 x H x W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="prediction_overlay_tiled.png")
    ap.add_argument("--patch_size", type=int, default=256,
                     help="Should match the patch_size used in prepare_dataset.py")
    ap.add_argument("--stride", type=int, default=192,
                     help="Smaller than patch_size = overlapping windows, smoother stitched result")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--base_features", type=int, default=32,
                     help="Must match the value used in train.py")
    ap.add_argument("--batch_size", type=int, default=8,
                     help="Lower this if you run out of RAM/VRAM")
    ap.add_argument("--no_legend", action="store_true",
                     help="Skip drawing the color legend box onto the output image")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Full native resolution -- no resize, matching how training patches were made.
    enhanced, fov, _ = enhance_image(img_rgb, target_size=None)

    model = UNet(in_channels=3, out_channels=5, base_features=args.base_features).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Image size: {enhanced.shape[1]}x{enhanced.shape[0]} -- tiling with "
          f"{args.patch_size}px patches, stride {args.stride}...")
    probs = sliding_window_predict(model, enhanced, fov, args.patch_size, args.stride,
                                    device, args.batch_size)

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

    if not args.no_legend:
        overlay_bgr = draw_legend(overlay_bgr)

    out_path = os.path.abspath(args.out)
    success = cv2.imwrite(out_path, overlay_bgr)
    if not success:
        raise RuntimeError(
            f"cv2.imwrite failed to save to '{out_path}'. "
            f"Check: is the folder writable? Does --out end in .png/.jpg? "
            f"Try a simpler path like --out C:\\Users\\ragha\\result.png"
        )
    print(f"Saved overlay to {out_path}")


if __name__ == "__main__":
    main()