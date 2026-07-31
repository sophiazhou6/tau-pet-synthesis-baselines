#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Rank grid-search configs by MEAN validation loss across all 5 folds (cross-validated
selection). Run after slurm/grid_search.slurm finishes.

Selection uses VALIDATION only — the 47-subject test set is never read here, so
picking the winner is unbiased. Report the winner's TEST metrics separately (train
it / it's already trained across folds → run generate + evaluate on the 47 test).
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import CHECKPOINT_DIR  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["atrophy", "ptau217"], default="atrophy")
    p.add_argument("--mask-mode", choices=["dk86", "wholebrain"], default="dk86")
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()
    run_tag = args.mode if args.mask_mode == "dk86" else f"{args.mode}_wholebrain"

    grid_root = os.path.join(CHECKPOINT_DIR, run_tag, "grid")
    cfgs = sorted(d for d in glob.glob(os.path.join(grid_root, "*")) if os.path.isdir(d))
    if not cfgs:
        sys.exit(f"No grid configs under {grid_root}/")

    rows = []
    for cfg_dir in cfgs:
        cfg = os.path.basename(cfg_dir)
        vals = []
        for f in range(args.n_folds):
            ck = os.path.join(cfg_dir, f"fold_{f}", "denseunet_best.pt")
            if os.path.exists(ck):
                vals.append(float(torch.load(ck, map_location="cpu")["val_loss"]))
        if vals:
            rows.append((cfg, float(np.mean(vals)), float(np.std(vals)), len(vals)))
    rows.sort(key=lambda r: r[1])

    print(f"\nGrid search — {run_tag}  (ranked by MEAN validation loss across folds)")
    print(f"{'config':26} {'mean_val':>10} {'std':>9} {'folds':>6}")
    print("-" * 54)
    for cfg, m, s, n in rows:
        flag = "  <-- best" if (cfg, m, s, n) == rows[0] else ""
        print(f"{cfg:26} {m:>10.5f} {s:>9.5f} {n:>6}{flag}")
    print(f"\nWINNER (by validation): {rows[0][0]}")
    print("Now report THIS config's TEST metrics (unbiased — test wasn't used to pick it):")
    print(f"  for f in 0 1 2 3 4; do")
    print(f"    python scripts/generate.py --mode {args.mode} --mask-mode {args.mask_mode} --fold $f \\")
    print(f"        --checkpoint results/checkpoints/{run_tag}/grid/{rows[0][0]}/fold_$f/denseunet_best.pt \\")
    print(f"        --out-dir results/generated/{run_tag}_tuned/fold_$f --overwrite")
    print(f"  done   # then suvr_sets.py / aggregate over the 5 folds")


if __name__ == "__main__":
    main()
