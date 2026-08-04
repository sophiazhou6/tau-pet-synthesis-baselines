#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Train the 3D Dense-U-Net baseline (MRI → tau PET, deterministic regression).

Mirrors the upstream training recipe (Adam, MSE loss) but on the TauGenNet lab
data pipeline at VOL_SHAPE, using the same train/val/test split as TauGenNet so
the result is a fair head-to-head baseline. Loss and outputs are masked to the
in-brain (DK86) region — the dataset already zeros the background, and we keep the
prediction zeroed there too (CLAUDE.md: always use the data mask that blocks out 0).

With --fold i (0-indexed), trains the i-th stratified CV fold of the mentor split:
the 47 heldout RIDs are the (never-trained) fixed test set and the 188 dev RIDs are
partitioned into 5 folds. Checkpoints are written under
results/checkpoints/<mode>/[fold_<i>/]. Without --fold, uses whatever split
build_dataloaders defaults to (the mentor split's single stratified dev split).

Never overwrites: refuses to start if the best checkpoint already exists unless
--overwrite is passed.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import (
    CHECKPOINT_DIR, build_dataloaders, LR, BATCH_SIZE, EPOCHS, PATIENCE, SEED, N_FOLDS,
)
from denseunet.masks import load_or_build_wholebrain_mask, inject_mask
from denseunet.model import DenseUNet3D


def masked_mse(pred, target, mask):
    """True in-brain MSE: mean squared error over DK86 voxels only.

    Unlike nn.MSELoss (which averages over all voxels including the zeroed
    background, diluting the loss and letting per-voxel gradient weight vary
    with each subject's brain-size fraction), this divides by the in-brain
    voxel count so the loss is undiluted and comparable across subjects.
    `mask` is the DK86 atlas mask — the same region evaluate_final.py scores on.
    """
    se = (pred - target) ** 2 * mask
    return se.sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, device, mask):
    model.eval()
    total, n = 0.0, 0
    for pet, mri, _cond in loader:
        pet, mri = pet.to(device), mri.to(device)
        total += masked_mse(model(mri), pet, mask).item() * pet.size(0)
        n += pet.size(0)
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser(description="Train Dense-U-Net baseline (MRI→tau)")
    p.add_argument("--mode", choices=["atrophy", "ptau217"], required=True,
                   help="Selects the matched-subject cohort/split (cond vector is ignored).")
    p.add_argument("--fold", type=int, default=None,
                   help="0-indexed CV fold of the mentor dev pool. None = single split.")
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--mask-mode", choices=["dk86", "wholebrain"], default="dk86",
                   help="dk86 = 86-region atlas mask; wholebrain = T1-seg brain mask (excl. background).")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="L2 weight decay. >0 uses AdamW (decoupled); 0 keeps plain Adam (baseline).")
    p.add_argument("--out-tag", type=str, default=None,
                   help="Extra subdir under results/checkpoints/<run_tag>/ to isolate grid-search runs.")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow overwriting an existing best checkpoint.")
    args = p.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Per-fold, per-mask-mode checkpoint dir so runs never clobber each other.
    run_tag = args.mode if args.mask_mode == "dk86" else f"{args.mode}_wholebrain"
    ckpt_dir = os.path.join(CHECKPOINT_DIR, run_tag)
    if args.out_tag:                       # grid-search: isolate each config
        ckpt_dir = os.path.join(ckpt_dir, args.out_tag)
    if args.fold is not None:
        ckpt_dir = os.path.join(ckpt_dir, f"fold_{args.fold}")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "denseunet_best.pt")
    if os.path.exists(best_path) and not args.overwrite:
        sys.exit(f"Refusing to overwrite existing checkpoint: {best_path} "
                 f"(pass --overwrite to replace).")

    fold_kwargs = {} if args.fold is None else {"fold_idx": args.fold, "n_folds": args.n_folds}
    train_ds, val_ds, test_ds, train_loader, val_loader, _ = build_dataloaders(
        mode=args.mode, batch_size=args.batch_size, use_dk_mask=True, **fold_kwargs)
    # Whole-brain variant: swap the dataset's shared DK86 mask for the T1-seg brain
    # mask so PET/MRI, loss, and metrics all use the whole brain (excl. background).
    if args.mask_mode == "wholebrain":
        inject_mask(load_or_build_wholebrain_mask(args.mode), train_ds, val_ds, test_ds)
    tag = run_tag if args.fold is None else f"{run_tag} fold {args.fold}/{args.n_folds}"
    print(f"[{tag}] {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    # Shared (1,H,W,D) mask — DK86 atlas or the injected whole-brain mask. Loss +
    # generation use this exact region so training, caching, and scoring align.
    dk_mask = train_ds._dk_mask.to(device)

    model = DenseUNet3D(in_ch=1, out_ch=1).to(device)
    if args.weight_decay > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.999), eps=1e-8)
    print(f"optimizer: {type(opt).__name__}  lr={args.lr}  wd={args.weight_decay}  "
          f"batch={args.batch_size}  epochs={args.epochs}  patience={args.patience}", flush=True)

    best_val = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for pet, mri, _cond in train_loader:
            pet, mri = pet.to(device), mri.to(device)
            opt.zero_grad()
            loss = masked_mse(model(mri), pet, dk_mask)
            loss.backward()
            opt.step()
            running += loss.item() * pet.size(0)
            n += pet.size(0)
        train_loss = running / max(n, 1)
        val_loss = evaluate(model, val_loader, device, dk_mask)

        flag = ""
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "mode": args.mode,
                        "fold": args.fold, "epoch": epoch, "val_loss": val_loss}, best_path)
            flag = "  *best"
        else:
            epochs_no_improve += 1

        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train {train_loss:.5f}  val {val_loss:.5f}{flag}", flush=True)

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no val improvement for {args.patience} epochs).")
            break

    print(f"Best val MSE: {best_val:.5f}  →  {best_path}")


if __name__ == "__main__":
    main()
