"""
prepare_dataset.py

Turns the raw IDRiD "A. Segmentation" folders into an enhanced,
UNet-ready dataset.

Expected raw IDRiD layout (this is how the official download is
structured -- adjust --raw_root if yours differs):

  <raw_root>/
    1. Original Images/
      a. Training Set/IDRiD_01.jpg ...
      b. Testing Set/IDRiD_55.jpg ...
    2. All Segmentation Groundtruths/
      a. Training Set/
        1. Microaneurysms/IDRiD_01_MA.tif
        2. Haemorrhages/IDRiD_01_HE.tif
        3. Hard Exudates/IDRiD_01_EX.tif
        4. Soft Exudates/IDRiD_01_SE.tif
        5. Optic Disc/IDRiD_01_OD.tif
      b. Testing Set/  (same sub-structure)

Not every image has every lesion type (e.g. many lack Soft Exudates) --
missing files are treated as an all-zero mask, which is correct: no
file means no instances of that lesion in that image.

TWO MODES:

  --mode full
      Resize whole image + masks to --size x --size (default 512).
      Fast, simple, gets a pipeline running end-to-end quickly. BUT:
      microaneurysms are only a few px wide at full 4288x2848
      resolution -- after a 512x512 resize most of them are gone.
      Good for a first working baseline / demo, weak for MA/HE recall.

  --mode patch
      Tiles each full-resolution image into overlapping patches
      (default 512x512, stride 384) BEFORE any resizing, so lesion
      pixels are preserved. Patches with zero lesion pixels are
      subsampled (--bg_keep_ratio) so the dataset isn't 95% empty
      background -- standard practice for IDRiD segmentation.
      This is the mode to use for real MA/HE/EX/SE segmentation
      quality. Produces far more training patches from the same 54
      training images.

Output layout (both modes):

  <out_root>/
    images/train/*.png          (enhanced RGB)
    images/val/*.png
    masks/train/<name>_MA.png   (0/255, one file per lesion channel)
    masks/train/<name>_HE.png
    masks/train/<name>_EX.png
    masks/train/<name>_SE.png
    masks/train/<name>_OD.png
    masks/val/...
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from enhancement import enhance_image

LESION_TYPES = ["MA", "HE", "EX", "SE", "OD"]
LESION_SUBDIRS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    "OD": "5. Optic Disc",
}


def find_image_dirs(raw_root, split_name):
    """split_name: 'Training Set' or 'Testing Set'"""
    img_dir = None
    for candidate in [
        Path(raw_root) / "1. Original Images" / f"a. {split_name}",
        Path(raw_root) / "1. Original Images" / f"b. {split_name}",
        Path(raw_root) / "1. Original Images" / split_name,
    ]:
        if candidate.exists():
            img_dir = candidate
            break
    if img_dir is None:
        raise FileNotFoundError(
            f"Could not find original images for '{split_name}' under {raw_root}. "
            f"Check --raw_root matches the IDRiD 'A. Segmentation' folder."
        )

    gt_root = Path(raw_root) / "2. All Segmentation Groundtruths"
    gt_dir = None
    for candidate in [gt_root / f"a. {split_name}", gt_root / f"b. {split_name}", gt_root / split_name]:
        if candidate.exists():
            gt_dir = candidate
            break
    if gt_dir is None:
        raise FileNotFoundError(
            f"Could not find groundtruths for '{split_name}' under {raw_root}."
        )
    return img_dir, gt_dir


def load_label_masks(gt_dir, stem, shape_hw):
    """Returns list of 5 uint8 HxW masks (0/1), in LESION_TYPES order."""
    masks = []
    for lesion in LESION_TYPES:
        subdir = gt_dir / LESION_SUBDIRS[lesion]
        found = None
        if subdir.exists():
            for ext in [".tif", ".tiff", ".png", ".jpg"]:
                candidate = subdir / f"{stem}_{lesion}{ext}"
                if candidate.exists():
                    found = candidate
                    break
        if found is not None:
            m = cv2.imread(str(found), cv2.IMREAD_GRAYSCALE)
            m = (m > 0).astype(np.uint8)
        else:
            m = np.zeros(shape_hw, dtype=np.uint8)
        masks.append(m)
    return masks


def save_sample(out_root, split, name, image_float01, masks01):
    img_dir = Path(out_root) / "images" / split
    mask_dir = Path(out_root) / "masks" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    img_uint8 = np.clip(image_float01 * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(img_dir / f"{name}.png"), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

    for lesion, m in zip(LESION_TYPES, masks01):
        cv2.imwrite(str(mask_dir / f"{name}_{lesion}.png"), (m * 255).astype(np.uint8))


def process_full(raw_root, out_root, split_name, out_split, size):
    img_dir, gt_dir = find_image_dirs(raw_root, split_name)
    image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    for path in tqdm(image_paths, desc=f"[full] {out_split}"):
        stem = path.stem
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        raw_masks = load_label_masks(gt_dir, stem, img_rgb.shape[:2])
        enhanced, fov, resized_masks = enhance_image(
            img_rgb, target_size=(size, size), label_masks=raw_masks
        )
        # zero out label pixels outside the FOV (consistent with image)
        resized_masks = [m * fov.astype(np.uint8) for m in resized_masks]
        save_sample(out_root, out_split, stem, enhanced, resized_masks)


def process_patch(raw_root, out_root, split_name, out_split, patch_size, stride, bg_keep_ratio, seed):
    rng = np.random.default_rng(seed)
    img_dir, gt_dir = find_image_dirs(raw_root, split_name)
    image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    total_saved, total_bg_skipped = 0, 0

    for path in tqdm(image_paths, desc=f"[patch] {out_split}"):
        stem = path.stem
        img_bgr = cv2.imread(str(path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        raw_masks = load_label_masks(gt_dir, stem, (H, W))

        # Enhance at full resolution first (no resize) so lesion pixels
        # are never shrunk away, then tile.
        enhanced, fov, _ = enhance_image(img_rgb, target_size=None, label_masks=None)
        stacked_masks = np.stack(raw_masks, axis=-1)  # H x W x 5

        y_steps = list(range(0, max(H - patch_size, 0) + 1, stride))
        x_steps = list(range(0, max(W - patch_size, 0) + 1, stride))
        if not y_steps:
            y_steps = [0]
        if not x_steps:
            x_steps = [0]

        patch_idx = 0
        for y in y_steps:
            for x in x_steps:
                y2, x2 = min(y + patch_size, H), min(x + patch_size, W)
                y1, x1 = y2 - patch_size, x2 - patch_size
                if y1 < 0 or x1 < 0:
                    continue  # image smaller than patch_size, skip

                fov_patch = fov[y1:y2, x1:x2]
                # skip patches mostly outside the retinal FOV (black corners)
                if fov_patch.mean() < 0.5:
                    continue

                img_patch = enhanced[y1:y2, x1:x2]
                mask_patch = stacked_masks[y1:y2, x1:x2, :]

                has_lesion = mask_patch.sum() > 0
                if not has_lesion:
                    if rng.random() > bg_keep_ratio:
                        total_bg_skipped += 1
                        continue

                name = f"{stem}_p{patch_idx:03d}"
                save_sample(
                    out_root, out_split, name, img_patch,
                    [mask_patch[:, :, i] for i in range(5)],
                )
                patch_idx += 1
                total_saved += 1

    print(f"[patch] {out_split}: saved {total_saved} patches, skipped {total_bg_skipped} empty-background patches")


def main():
    ap = argparse.ArgumentParser(description="Prepare enhanced IDRiD segmentation dataset for UNet")
    ap.add_argument("--raw_root", required=True,
                     help="Path to the IDRiD 'A. Segmentation' folder")
    ap.add_argument("--out_root", required=True, help="Where to write the prepared dataset")
    ap.add_argument("--mode", choices=["full", "patch"], default="patch")
    ap.add_argument("--size", type=int, default=512, help="[full mode] resize target (size x size)")
    ap.add_argument("--patch_size", type=int, default=512, help="[patch mode] patch height/width")
    ap.add_argument("--stride", type=int, default=384, help="[patch mode] sliding window stride")
    ap.add_argument("--bg_keep_ratio", type=float, default=0.15,
                     help="[patch mode] fraction of empty-background patches to keep")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.mode == "full":
        process_full(args.raw_root, args.out_root, "Training Set", "train", args.size)
        process_full(args.raw_root, args.out_root, "Testing Set", "val", args.size)
    else:
        process_patch(args.raw_root, args.out_root, "Training Set", "train",
                       args.patch_size, args.stride, args.bg_keep_ratio, args.seed)
        process_patch(args.raw_root, args.out_root, "Testing Set", "val",
                       args.patch_size, args.stride, 1.0, args.seed)  # keep all val patches

    print(f"Done. Prepared dataset written to {args.out_root}")


if __name__ == "__main__":
    main()
