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
    use_film = getattr(model, "film_cond_dim", 0) > 0
    for pet, mri, cond in loader:
        pet, mri = pet.to(device), mri.to(device)
        pred = model(mri, cond=cond.to(device)) if use_film else model(mri)
        total += masked_mse(pred, pet, mask).item() * pet.size(0)
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
    p.add_argument("--val-frac", type=float, default=0.16,
                   help="Val fraction of dev pool; 0 = no validation, train on full dev pool.")
    p.add_argument("--use-suvr", action="store_true",
                   help="Paint regional SUVR as a 2nd input channel alongside MRI.")
    p.add_argument("--use-suvr-film", action="store_true",
                   help="Inject regional SUVR via FiLM conditioning at the bottleneck (alternative to --use-suvr).")
    p.add_argument("--use-phase6-controls", action="store_true",
                   help="Train on the pooled CN+MCI+AD phase6 cohort (plain MRI-only, no SUVR channel).")
    p.add_argument("--phase6-fold", type=int, default=0,
                   help="Which phase6 manifest fold's test-set to use as the fixed held-out test.")
    p.add_argument("--mentor-split-dir", type=str, default=None,
                   help="Override directory to read heldout_test_split.csv/train_val_split.csv from, "
                        "isolated from the shared data/raw/ root.")
    p.add_argument("--use-controls-322", action="store_true",
                   help="Train on the 322-subject controls_322 cohort (CN+MCI+AD). Compatible with --use-suvr/--use-suvr-film, unlike --use-phase6-controls.")
    p.add_argument("--out-tag", type=str, default=None,
                   help="Extra subdir under results/checkpoints/<run_tag>/ to isolate grid-search runs.")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow overwriting an existing best checkpoint.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Load model weights from this checkpoint (model only, not optimizer state) "
                        "and continue training from its saved epoch + 1.")
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
    assert not (args.use_suvr and args.use_suvr_film), "--use-suvr and --use-suvr-film are mutually exclusive"
    assert not (args.use_phase6_controls and (args.use_suvr or args.use_suvr_film)), \
        "--use-phase6-controls is plain MRI-only (controls have no regional SUVR) — cannot combine with --use-suvr/--use-suvr-film"
    assert not (args.use_controls_322 and args.use_phase6_controls), \
        "--use-controls-322 and --use-phase6-controls are mutually exclusive controls-cohort mechanisms"
    train_ds, val_ds, test_ds, train_loader, val_loader, _ = build_dataloaders(
        mode=args.mode, batch_size=args.batch_size, use_dk_mask=True,
        val_frac=args.val_frac, suvr_input=args.use_suvr, suvr_as_cond=args.use_suvr_film,
        use_phase6_manifest=args.use_phase6_controls, phase6_fold=args.phase6_fold, mentor_split_dir=args.mentor_split_dir,
        use_controls_322=args.use_controls_322,
        **fold_kwargs)
    # Whole-brain variant: swap the dataset's shared DK86 mask for the T1-seg brain
    # mask so PET/MRI, loss, and metrics all use the whole brain (excl. background).
    if args.mask_mode == "wholebrain":
        inject_mask(load_or_build_wholebrain_mask(args.mode), train_ds, val_ds, test_ds)
    tag = run_tag if args.fold is None else f"{run_tag} fold {args.fold}/{args.n_folds}"
    print(f"[{tag}] {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    # Shared (1,H,W,D) mask — DK86 atlas or the injected whole-brain mask. Loss +
    # generation use this exact region so training, caching, and scoring align.
    dk_mask = train_ds._dk_mask.to(device)

    _paint = os.environ.get("TAUGENNET_PAINT", "suvr").lower()
    _extra = 2 if _paint == "both" else 1
    model = DenseUNet3D(in_ch=(1 + _extra) if args.use_suvr else 1, out_ch=1, film_cond_dim=86 if args.use_suvr_film else 0).to(device)
    print(f"[paint={_paint}] DenseUNet in_ch={(1 + _extra) if args.use_suvr else 1}")
    if args.weight_decay > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               betas=(0.9, 0.999), eps=1e-8)
    print(f"optimizer: {type(opt).__name__}  lr={args.lr}  wd={args.weight_decay}  "
          f"batch={args.batch_size}  epochs={args.epochs}  patience={args.patience}", flush=True)

    start_epoch = 1
    if args.resume_from:
        rckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(rckpt["model"])
        start_epoch = rckpt["epoch"] + 1
        print(f"Resumed weights from {args.resume_from} (was at epoch {rckpt['epoch']}) "
              f"-> continuing from epoch {start_epoch}", flush=True)

    best_val = float("inf")
    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        for pet, mri, cond in train_loader:
            pet, mri = pet.to(device), mri.to(device)
            opt.zero_grad()
            pred = model(mri, cond=cond.to(device)) if args.use_suvr_film else model(mri)
            loss = masked_mse(pred, pet, dk_mask)
            loss.backward()
            opt.step()
            running += loss.item() * pet.size(0)
            n += pet.size(0)
        train_loss = running / max(n, 1)

        if len(val_loader) == 0:
            torch.save({"model": model.state_dict(), "mode": args.mode,
                        "fold": args.fold, "epoch": epoch, "val_loss": None}, best_path)
            print(f"epoch {epoch:3d}/{args.epochs}  train {train_loss:.5f}  (no validation)", flush=True)
            continue

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
