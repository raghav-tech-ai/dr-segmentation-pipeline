# DR Lesion Segmentation (IDRiD + UNet)

Enhances the IDRiD "A. Segmentation" dataset using a Python port of your
`doctor_upload_pipeline` (MATLAB) enhancement chain, then trains a UNet to
segment 5 classes: **Microaneurysms (MA), Haemorrhages (HE), Hard Exudates
(EX), Soft Exudates (SE), Optic Disc (OD)**.

## Why this isn't a direct reuse of the MATLAB zip

`doctor_upload_pipeline` resizes to 512×512 *before* enhancement and only
tracks a FOV mask — it was built for single-image quality gating, not for
keeping pixel-level lesion labels aligned. `enhancement.py` here reproduces
the same illumination-normalize → CLAHE → denoise chain (so training-time
preprocessing matches what the doctor-upload path will eventually do at
inference time) but resizes/tiles the label masks in lockstep with the
image using nearest-neighbour interpolation.

## Two dataset-prep modes — read this before picking one

| | `full` | `patch` (default, recommended) |
|---|---|---|
| What it does | Resizes whole image + masks to 512×512 | Tiles full-res image into 512×512 patches (stride 384) before any resizing |
| MA/HE recall | Poor — microaneurysms are ~2-5px in the original 4288×2848 image and mostly vanish after a 512² resize | Preserved — no resizing before tiling |
| Speed to first result | Fast, ~55 training images | Slower — but generates hundreds of patches from the same 54 images |
| When to use | Quick pipeline smoke-test / demo fallback | Actual model you'd report Dice/IoU numbers for |

Start with `patch` mode. Only fall back to `full` if you need something
running in the next hour for a demo and can live with weak MA/HE detection
(EX/SE/OD are large enough to survive `full` mode reasonably well).

---

## 1. Setup (one-time)

```bash
# from inside dr_segmentation_pipeline/
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If you don't have a GPU (most laptops don't), the default `pip install torch`
gets you a CPU build automatically — training will just be slower. `train.py`
auto-detects CUDA, Apple Silicon (MPS), or falls back to CPU.

## 2. Download IDRiD

Get the **"A. Segmentation"** folder from the official IDRiD dataset
(50-ish training + val images with pixel-level lesion masks — this is the
small subset with segmentation ground truth, not the full ~500-image
grading set). Unzip it anywhere; note the path.

Expected structure `prepare_dataset.py` looks for:
```
A. Segmentation/
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
    b. Testing Set/ (same layout)
```
If your download uses different casing/spacing, either rename the folders
to match or edit the path-matching lists in `find_image_dirs()` in
`prepare_dataset.py`.

## 3. Enhance + prepare the dataset

```bash
python prepare_dataset.py \
  --raw_root "/path/to/A. Segmentation" \
  --out_root ./prepared_dataset \
  --mode patch \
  --patch_size 512 \
  --stride 384 \
  --bg_keep_ratio 0.15
```

This writes `prepared_dataset/images/{train,val}/*.png` (enhanced RGB) and
`prepared_dataset/masks/{train,val}/<name>_<LESION>.png` (binary masks).
Expect this to take a few minutes on a laptop CPU — non-local-means
denoising is the slow step. Runtime scales with image count × patches;
if it's too slow, raise `--stride` (fewer patches) or start with `--mode full`
to sanity-check the rest of the pipeline first.

## 4. Train

```bash
python train.py \
  --data_root ./prepared_dataset \
  --epochs 40 \
  --batch_size 4 \
  --base_features 32
```

Laptop-specific notes:
- **Out of RAM/VRAM?** Drop `--base_features` to 16, and/or `--batch_size` to 2.
- **CPU-only training is slow.** ~50-200 patches/epoch on a 4-core laptop CPU
  is realistically 5-20 min/epoch depending on patch count and `base_features`.
  For a hackathon timeline, consider training on Google Colab's free GPU tier
  instead (upload `prepared_dataset/`, same `train.py` runs unmodified) — you
  already have free cloud credits per the team's resource notes.
- Checkpoints save to `./checkpoints/last.pt` (every epoch) and `best.pt`
  (best val loss so far) — `--resume checkpoints/last.pt` continues training
  after an interruption.
- The training log prints per-lesion Dice score each epoch. Expect MA/HE Dice
  to start near 0 and climb slowly — they're the hardest classes by far
  (smallest, rarest). EX and OD should climb fastest.

## 5. Run inference on a new image

```bash
python predict.py \
  --image path/to/some_fundus.jpg \
  --checkpoint checkpoints/best.pt \
  --base_features 32 \
  --out prediction_overlay.png
```

Produces a color-coded overlay (red=MA, orange=HE, yellow=EX, cyan=SE,
magenta=OD) and prints flagged pixel counts per lesion type.

## 6. Sanity-check before trusting results

- With only ~54 training images (even split into hundreds of patches from
  the same underlying photos), expect real overfitting risk. Watch the
  train vs. val loss gap in the epoch log.
- Report Dice **per lesion class**, not one averaged number — MA performance
  will legitimately be much lower than EX/OD, and averaging hides that.
- If you're integrating this into the RuralRetinal AI doctor portal
  (Grad-CAM explainability layer), note this is a *separate* model from
  the EfficientNet-B0 DR-grading classifier — segmentation gives you lesion
  *localization*, Grad-CAM on EfficientNet gives you classifier *attention*.
  Decide whether the demo needs both or whether one clearly communicates the
  "why" to a non-specialist rural health worker; running two models adds
  inference time and deployment complexity for the MVP.
