#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Post-CV aggregation — run after the 5-fold array finishes. Produces the
mentor-facing deliverables:

  1. CV mean ± std tables (whole-brain DK86 + the 3 SUVR R SETs) via the existing
     aggregate_cv_eval.py / aggregate_suvr_sets.py.
  2. An ENSEMBLE prediction set (mean of the 5 folds' predictions per subject on
     the fixed 47 heldout test), scored to produce:
       - the 4 SUVR figures (SET 1 scatter, SET 2/3 region bars, pooled-vs-xsub)
       - (dk86 only) the glass-brain / slice / scatter figures via evaluate_final
  3. CV per-region bar charts with fold error bars (mean ± std over folds).

Reuses the existing scripts by subprocess so nothing is reimplemented.

Usage:
    python scripts/aggregate_cv.py --mode atrophy --mask-mode dk86
    python scripts/aggregate_cv.py --mode atrophy --mask-mode wholebrain
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from denseunet.config import GENERATED_DIR, REPO_ROOT, TAUGENNET_ROOT  # noqa: E402
from denseunet import figs  # noqa: E402

PY = "/home/sz3962/.conda/envs/taugennet/bin/python3"
RECORDS_DIR = os.path.join(REPO_ROOT, "results", "records")
FIGURES_DIR = os.path.join(REPO_ROOT, "results", "figures")


def _run(cmd):
    env = dict(os.environ, TAUGENNET_ROOT=TAUGENNET_ROOT, CUDA_VISIBLE_DEVICES="")
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def build_ensemble(run_tag, n_folds):
    """Mean of the folds' predictions per subject → generated/<run_tag>/ensemble/."""
    gen_base = os.path.join(GENERATED_DIR, run_tag)
    fold_dirs = [os.path.join(gen_base, f"fold_{i}") for i in range(n_folds)]
    fold_dirs = [d for d in fold_dirs if os.path.isdir(d)]
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(fold_dirs[0], "subject_*.npy")))
    ens_dir = os.path.join(gen_base, "ensemble")
    os.makedirs(ens_dir, exist_ok=True)
    for name in names:
        stack = np.stack([np.load(os.path.join(d, name)) for d in fold_dirs], axis=0)
        np.save(os.path.join(ens_dir, name), stack.mean(0).astype(np.float32))
    print(f"[ensemble] {len(names)} subjects × {len(fold_dirs)} folds → {ens_dir}")
    return ens_dir


def wholebrain_glass_brain(mode, ens_dir, run_tag, n_folds):
    """Render glass/slice/scatter figures for the whole-brain ensemble.

    evaluate_final.py is DK86-hardcoded, but glass_brain_final.run_mode(data=...)
    just renders whatever volumes it's handed — so we build whole-brain-masked
    ensemble volumes and call it directly. No DK86 tables are produced (only the
    glass brains), which is exactly what's wanted for the whole-brain variant.
    """
    for pth in (TAUGENNET_ROOT, os.path.join(TAUGENNET_ROOT, "scripts")):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    from glass_brain_final import run_mode  # noqa: E402
    from src.dataset_final import unnormalize  # noqa: E402
    from denseunet.config import build_dataloaders  # noqa: E402
    from denseunet.masks import load_or_build_wholebrain_mask, inject_mask  # noqa: E402

    _, _, test_ds, *_ = build_dataloaders(mode=mode, use_dk_mask=True)
    mask = inject_mask(load_or_build_wholebrain_mask(mode), test_ds)
    mask_np = (mask.squeeze(0).numpy() > 0)

    real_suvr, gen_suvr, real_norm, gen_masked = [], [], [], []
    for i in range(len(test_ds)):
        real = test_ds[i][0].squeeze().numpy()               # normalized, wb-masked
        gen = np.load(os.path.join(ens_dir, f"subject_{i:03d}.npy")).astype(np.float32)
        pmin, pmax = test_ds.get_pet_norms(i)
        real_norm.append(real)
        gen_masked.append(gen * mask_np)
        real_suvr.append(unnormalize(real, pmin, pmax))
        gen_suvr.append(unnormalize(gen, pmin, pmax) * mask_np)
    gb_gen_suvr = [g * (r > 0) for g, r in zip(gen_suvr, real_norm)]  # display mask = whole brain

    gb_dir = os.path.join(FIGURES_DIR, run_tag, "glass_brain")
    run_mode(mode, "silu", gb_dir, use_dk_mask=True,
             data=(real_suvr, gb_gen_suvr, real_norm, gen_masked, test_ds))
    print(f"[wholebrain glass brain] → {gb_dir}")


def cv_errorbar_bars(run_tag, mode):
    """CV per-region bars with fold error bars, from aggregate_suvr_sets' *_cv.csv."""
    cv_dir = os.path.join(RECORDS_DIR, f"cv_{run_tag}")
    fig_dir = os.path.join(FIGURES_DIR, run_tag, "cv")
    for setname, csv, title in [
        ("set2", "region_pooled_SUVR_cv.csv",
         f"SET 2 — Pooled-voxel Pearson R per region ({run_tag}, 5-fold CV)"),
        ("set3", "region_crosssubject_SUVR_cv.csv",
         f"SET 3 — Regional cross-subject Pearson R ({run_tag}, 5-fold CV)")]:
        path = os.path.join(cv_dir, csv)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        figs.plot_region_bars_ci(df["R_mean"], df["R_std"], title,
                                 os.path.join(fig_dir, f"{setname}_region_R_SUVR_cv.png"))
    print(f"[cv bars] → {fig_dir}")


def main():
    p = argparse.ArgumentParser(description="Aggregate a finished 5-fold CV run.")
    p.add_argument("--mode", choices=["atrophy", "ptau217"], required=True)
    p.add_argument("--mask-mode", choices=["dk86", "wholebrain"], default="dk86")
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()
    run_tag = args.mode if args.mask_mode == "dk86" else f"{args.mode}_wholebrain"

    # 1. CV mean ± std tables
    _run([PY, os.path.join(TAUGENNET_ROOT, "scripts", "aggregate_cv_eval.py"),
          os.path.join(RECORDS_DIR, f"cv_{run_tag}")])
    _run([PY, "scripts/aggregate_suvr_sets.py", "--mode", args.mode, "--mask-mode", args.mask_mode])

    # 2. Ensemble prediction set + its figures
    ens_dir = build_ensemble(run_tag, args.n_folds)
    if args.mask_mode == "dk86":
        # evaluate_final: glass-brain / slice / regional figures (DK86 only)
        _run([PY, "scripts/evaluate.py", "--mode", args.mode, "--generated-dir", ens_dir])
    else:
        # whole-brain: render glass brains directly (skip DK86 tables)
        wholebrain_glass_brain(args.mode, ens_dir, run_tag, args.n_folds)
    # the 4 SUVR figures for the ensemble
    _run([PY, "scripts/suvr_sets.py", "--mode", args.mode, "--mask-mode", args.mask_mode,
          "--generated-dir", ens_dir])

    # 3. CV per-region error-bar bars
    cv_errorbar_bars(run_tag, args.mode)

    gb = os.path.join(FIGURES_DIR, args.mode, "glass_brain") if args.mask_mode == "dk86" \
        else os.path.join(FIGURES_DIR, run_tag, "glass_brain")
    print(f"\n=== aggregation done ({run_tag}) ===")
    print(f"  CV tables    → {RECORDS_DIR}/cv_{run_tag}/")
    print(f"  SUVR figures → {FIGURES_DIR}/{run_tag}/suvr/  (ensemble) + /{run_tag}/cv/ (fold error bars)")
    print(f"  Glass brain  → {gb}/  (ensemble)")


if __name__ == "__main__":
    main()
