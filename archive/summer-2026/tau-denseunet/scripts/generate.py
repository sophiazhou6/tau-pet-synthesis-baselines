#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Generate Dense-U-Net predictions for the test set and cache them in the exact
format TauGenNet's evaluate_final.py expects: one subject_{i:03d}.npy per test
subject, holding the predicted tau PET in normalized [0,1] space, indexed by
position in test_ds.

Because we build the test set with the same build_dataloaders(mode=...) defaults
(split + SEED) that evaluate_final.py uses, the ordering matches and the cached
files can be scored directly with:

    cd $TAUGENNET_ROOT
    python scripts/evaluate_final.py --mode <mode> --use-cached \\
        --generated-dir <this repo>/results/generated/<mode>

Outputs are written to results/generated/<mode>/. Never overwrites an existing
non-empty directory unless --overwrite is given.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import CHECKPOINT_DIR, GENERATED_DIR, build_dataloaders, N_FOLDS
from denseunet.masks import load_or_build_wholebrain_mask, inject_mask
from denseunet.model import DenseUNet3D


def main():
    p = argparse.ArgumentParser(description="Cache Dense-U-Net test predictions")
    p.add_argument("--mode", choices=["atrophy", "ptau217"], required=True)
    p.add_argument("--fold", type=int, default=None,
                   help="0-indexed CV fold. Defaults the checkpoint/out dirs to "
                        "<...>/<run_tag>/fold_<i>/. The mentor test set (47) is identical "
                        "across folds; each fold's model predicts it.")
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--mask-mode", choices=["dk86", "wholebrain"], default="dk86",
                   help="dk86 = 86-region atlas mask; wholebrain = T1-seg brain mask.")
    p.add_argument("--use-suvr", action="store_true",
                   help="Model expects a painted regional-SUVR 2nd input channel.")
    p.add_argument("--use-suvr-film", action="store_true",
                   help="Model expects FiLM conditioning on regional SUVR (alternative to --use-suvr).")
    p.add_argument("--split", choices=["test", "val"], default="test",
                   help="Generate against the held-out test set (default) or the validation split.")
    p.add_argument("--mentor-split-dir", type=str, default=None,
                   help="Override directory to read heldout_test_split.csv/train_val_split.csv from, "
                        "isolated from the shared data/raw/ root.")
    p.add_argument("--use-controls-322", action="store_true",
                   help="Use the 322-subject controls_322 cohort (CN+MCI+AD). MUST match how the "
                        "checkpoint was trained, or cached predictions map to the wrong subjects.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Defaults to results/checkpoints/<run_tag>/[fold_<i>/]denseunet_best.pt")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Defaults to results/generated/<run_tag>/[fold_<i>/]")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_tag = args.mode if args.mask_mode == "dk86" else f"{args.mode}_wholebrain"
    fold_sub = "" if args.fold is None else f"fold_{args.fold}"
    ckpt = args.checkpoint or os.path.join(
        CHECKPOINT_DIR, run_tag, fold_sub, "denseunet_best.pt")
    out_dir = args.out_dir or os.path.join(GENERATED_DIR, run_tag, fold_sub)

    if os.path.isdir(out_dir) and any(f.endswith(".npy") for f in os.listdir(out_dir)) \
            and not args.overwrite:
        sys.exit(f"Refusing to overwrite cached generations in {out_dir} "
                 f"(pass --overwrite).")
    os.makedirs(out_dir, exist_ok=True)

    # Same fold + defaults as the scorer → identical test-set ordering.
    assert not (args.use_suvr and args.use_suvr_film), "--use-suvr and --use-suvr-film are mutually exclusive"
    fold_kwargs = {} if args.fold is None else {"fold_idx": args.fold, "n_folds": args.n_folds}
    train_ds, val_ds, test_ds, *_ = build_dataloaders(
        mode=args.mode, use_dk_mask=True, val_frac=0.0,
        suvr_input=args.use_suvr, suvr_as_cond=args.use_suvr_film,
        mentor_split_dir=args.mentor_split_dir,
        use_controls_322=args.use_controls_322, **fold_kwargs)
    if args.split == "val":
        test_ds = val_ds
    if args.mask_mode == "wholebrain":
        inject_mask(load_or_build_wholebrain_mask(args.mode), train_ds, val_ds, test_ds)
    tag = run_tag if args.fold is None else f"{run_tag} fold {args.fold}/{args.n_folds}"
    print(f"[{tag}] test subjects: {len(test_ds)}")

    # Shared (1,H,W,D) mask — DK86 atlas or injected whole-brain — the region scored.
    dk_mask = test_ds._dk_mask.to(device)

    _paint = os.environ.get("TAUGENNET_PAINT", "suvr").lower()
    _extra = 2 if _paint == "both" else 1
    model = DenseUNet3D(in_ch=(1 + _extra) if args.use_suvr else 1, out_ch=1, film_cond_dim=86 if args.use_suvr_film else 0).to(device)
    print(f"[paint={_paint}] DenseUNet in_ch={(1 + _extra) if args.use_suvr else 1}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    _vl = state.get('val_loss')
    _vl_str = f"{_vl:.5f}" if _vl is not None else "n/a (no validation)"
    print(f"Loaded checkpoint: {ckpt}  (epoch {state.get('epoch')}, val {_vl_str})")

    with torch.no_grad():
        for i in range(len(test_ds)):
            pet, mri, cond = test_ds[i]
            mri_b = mri.unsqueeze(0).to(device)        # (1,in_ch,H,W,D)
            cond_b = cond.unsqueeze(0).to(device) if args.use_suvr_film else None
            out = model(mri_b, cond=cond_b) if args.use_suvr_film else model(mri_b)
            pred = (out * dk_mask).squeeze().cpu().numpy().astype(np.float32)
            # No clipping: the linear output head can exceed [0,1]; keep raw
            # predictions so over/under-shoot is visible and eval stays honest.
            np.save(os.path.join(out_dir, f"subject_{i:03d}.npy"), pred)

    print(f"Wrote {len(test_ds)} files → {out_dir}")
    fold_flag = "" if args.fold is None else f" --fold {args.fold}"
    print(f"Score with:  python scripts/suvr_sets.py --mode {args.mode}{fold_flag} "
          f"--mask-mode {args.mask_mode}")


if __name__ == "__main__":
    main()
