"""
enhancement.py

Python port of the MATLAB doctor_upload_pipeline (CORE_FUNCTIONS/*.m).
Kept behaviourally close to the original so that train-time preprocessing
matches what the doctor-upload inference pipeline will eventually do:

    resize -> illumination normalize -> CLAHE -> denoise -> mask apply

Key difference from the MATLAB version: every function here optionally
takes a `mask` (or a list of label masks) that gets resized IN SYNC with
the image, using nearest-neighbour interpolation, so segmentation labels
never drift out of pixel alignment with the enhanced image.
"""

import cv2
import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------
# 1. FOV mask  (port of CORE_FUNCTIONS/getFOVMask.m)
# ---------------------------------------------------------------------
def get_fov_mask(img_float):
    """
    img_float: HxWx3 float image in [0, 1]
    returns: HxW bool mask of the visible retinal field-of-view
    """
    H, W = img_float.shape[:2]
    image_area = H * W

    gray = cv2.cvtColor(img_float.astype(np.float32), cv2.COLOR_RGB2GRAY)
    g_min, g_max = gray.min(), gray.max()
    if g_max <= g_min:
        return np.ones((H, W), dtype=bool)
    gray_norm = (gray - g_min) / (g_max - g_min)

    low_p = np.percentile(gray_norm, 10)
    high_p = np.percentile(gray_norm, 90)
    threshold = low_p + 0.20 * (high_p - low_p)
    threshold = float(np.clip(threshold, 0.04, 0.30))

    mask = gray_norm > threshold

    min_area = max(int(round(0.01 * image_area)), 500)
    mask = _remove_small_objects(mask, min_area)

    labeled, n = ndimage.label(mask)
    if n == 0:
        return _fallback_mask(H, W)

    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    largest_idx = int(np.argmax(sizes)) + 1
    largest_size = sizes[largest_idx - 1]
    mask = labeled == largest_idx

    mask = ndimage.binary_fill_holes(mask)

    disk12 = _disk(12)
    disk5 = _disk(5)
    mask = ndimage.binary_closing(mask, structure=disk12)
    mask = ndimage.binary_opening(mask, structure=disk5)
    mask = ndimage.binary_fill_holes(mask)

    labeled, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        largest_idx = int(np.argmax(sizes)) + 1
        mask = labeled == largest_idx

    area_ratio = mask.sum() / image_area
    if largest_size < 0.05 * image_area or area_ratio < 0.10:
        return _fallback_mask(H, W)

    ys, xs = np.where(mask)
    if ys.size == 0:
        return _fallback_mask(H, W)
    bbox_w = (xs.max() - xs.min() + 1) / W
    bbox_h = (ys.max() - ys.min() + 1) / H
    if bbox_w < 0.25 or bbox_h < 0.25:
        return _fallback_mask(H, W)

    mask = ndimage.binary_fill_holes(mask)
    return mask


def _remove_small_objects(mask, min_area):
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_area
    return keep[labeled]


def _disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x ** 2 + y ** 2) <= radius ** 2


def _fallback_mask(H, W):
    y, x = np.ogrid[:H, :W]
    cx, cy = (W - 1) / 2, (H - 1) / 2
    rx, ry = 0.47 * W, 0.47 * H
    return ((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2) <= 1


# ---------------------------------------------------------------------
# 2. Illumination normalization (port of illuminationNormalize.m)
# ---------------------------------------------------------------------
def illumination_normalize(img_float, mask):
    sigma = max(5, round(min(img_float.shape[0], img_float.shape[1]) / 20))
    ksize = int(2 * round(3 * sigma) + 1)  # odd kernel covering ~3 sigma

    out = np.zeros_like(img_float)
    for c in range(3):
        ch = img_float[:, :, c]
        bg = cv2.GaussianBlur(ch, (ksize, ksize), sigma)
        mean_bg = bg[mask].mean() if mask.any() else bg.mean()
        corrected = ch - bg + mean_bg
        corrected[~mask] = 0
        out[:, :, c] = corrected
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------
# 3. CLAHE on luminance only (port of applyCLAHE.m)
# ---------------------------------------------------------------------
def apply_clahe(img_float, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    MATLAB used adapthisteq('ClipLimit', 0.01, 'Distribution', 'rayleigh')
    on the LAB L channel. OpenCV's CLAHE uses a differently-scaled clip
    limit and a uniform (not rayleigh) redistribution -- functionally
    equivalent contrast enhancement, tuned clip_limit=2.0 to match the
    MATLAB pipeline's visual strength on fundus images.
    """
    img_uint8 = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return out.astype(np.float32) / 255.0


# ---------------------------------------------------------------------
# 4. Denoise (port of denoiseImage.m)
# ---------------------------------------------------------------------
def denoise_image(img_float, h=6):
    """
    MATLAB used non-local means per channel (imnlmfilt). OpenCV's
    fastNlMeansDenoisingColored operates on all 3 channels together and
    is much faster -- practically equivalent result, far better suited
    to a laptop CPU.
    """
    img_uint8 = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(
        img_uint8, None, h=h, hColor=h, templateWindowSize=5, searchWindowSize=11
    )
    return denoised.astype(np.float32) / 255.0


# ---------------------------------------------------------------------
# 5. Full enhancement chain (port of enhanceImage.m)
#    -- extended to resize an arbitrary list of label masks in sync.
# ---------------------------------------------------------------------
def enhance_image(img, target_size=(512, 512), label_masks=None):
    """
    img: HxWx3 uint8 or float image (RGB)
    target_size: (H, W) to resize to. Pass None to skip resizing
                 entirely (use this for the patch-based pipeline, where
                 tiling happens BEFORE enhancement so nothing shrinks).
    label_masks: optional list of HxW uint8/bool arrays (segmentation
                 ground truth, one per lesion type) resized in lockstep
                 with the image using nearest-neighbour interpolation.

    Returns (enhanced_image_float01, fov_mask, resized_label_masks)
    """
    if img.dtype != np.float32 and img.dtype != np.float64:
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = img.astype(np.float32)

    if target_size is not None:
        th, tw = target_size
        img_r = cv2.resize(img_f, (tw, th), interpolation=cv2.INTER_AREA)
        resized_labels = None
        if label_masks is not None:
            resized_labels = [
                cv2.resize(m.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
                for m in label_masks
            ]
    else:
        img_r = img_f
        resized_labels = label_masks

    fov = get_fov_mask(img_r)

    norm = illumination_normalize(img_r, fov)
    clahe = apply_clahe(norm)
    denoised = denoise_image(clahe)

    mask3 = np.repeat(fov[:, :, None], 3, axis=2)
    denoised[~mask3] = 0

    enhanced = np.clip(denoised, 0, 1)
    return enhanced, fov, resized_labels
