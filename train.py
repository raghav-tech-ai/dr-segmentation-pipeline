"""
train.py -- Trains the UNet on the prepared dataset.

Usage (after running prepare_dataset.py):
    python train.py --data_root ./prepared_dataset --epochs 40 --batch_size 4
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import IDRiDSegDataset, LESION_TYPES
from losses import BCEDiceLoss, per_class_dice
from model import UNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    class_dice_sums = {k: 0.0 for k in LESION_TYPES}
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            batch_dice = per_class_dice(logits, masks)
            for k in LESION_TYPES:
                class_dice_sums[k] += batch_dice[k]
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_dice = {k: v / max(n_batches, 1) for k, v in class_dice_sums.items()}
    return avg_loss, avg_dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Output of prepare_dataset.py")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base_features", type=int, default=32,
                     help="Lower this (e.g. 16) if you run out of RAM/VRAM on a laptop")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--checkpoint_dir", default="./checkpoints")
    ap.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_ds = IDRiDSegDataset(args.data_root, "train", augment=True)
    val_ds = IDRiDSegDataset(args.data_root, "val", augment=False)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = UNet(in_channels=3, out_channels=5, base_features=args.base_features).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    start_epoch = 0
    best_val_loss = float("inf")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_dice = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_dice = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)
        dt = time.time() - t0

        dice_str = " ".join(f"{k}={val_dice[k]:.3f}" for k in LESION_TYPES)
        print(f"Epoch {epoch+1}/{args.epochs} | {dt:.0f}s | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | val_dice: {dice_str}")

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_dice": val_dice,
            "best_val_loss": best_val_loss,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    main()
