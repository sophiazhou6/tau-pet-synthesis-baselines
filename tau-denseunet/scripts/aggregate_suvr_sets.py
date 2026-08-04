#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Aggregate per-fold SUVR SET 1/2/3 Pearson R (from scripts/suvr_sets.py) into the
CV result: mean ± std across folds.

Reads results/records/cv_<mode>/fold_*/suvr_sets_fold*.json and writes:
  results/records/cv_<mode>/suvr_sets_cv_summary.json   — all aggregates
  results/records/cv_<mode>/region_pooled_SUVR_cv.csv   — SET 2: per-region mean,std (86)
  results/records/cv_<mode>/region_crosssubject_SUVR_cv.csv — SET 3: per-region mean,std (86)

Reported in two ways:
  * fold-level overall: mean ± std of each fold's overall R (across the 5 folds)
  * per-region: mean ± std of each region's R across the 5 folds

Usage:
    python scripts/aggregate_suvr_sets.py --mode atrophy
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import REPO_ROOT  # noqa: E402

RECORDS_DIR = os.path.join(REPO_ROOT, "results", "records")


def _ms(vals):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    return (float(a.mean()) if n else float("nan"),
            float(a.std(ddof=0)) if n > 1 else 0.0, n)


def main():
    p = argparse.ArgumentParser(description="Aggregate per-fold SUVR SET 1/2/3 R across folds.")
    p.add_argument("--mode", choices=["atrophy", "ptau217"], required=True)
    p.add_argument("--mask-mode", choices=["dk86", "wholebrain"], default="dk86")
    p.add_argument("--cv-dir", default=None,
                   help="Defaults to results/records/cv_<run_tag>.")
    args = p.parse_args()

    run_tag = args.mode if args.mask_mode == "dk86" else f"{args.mode}_wholebrain"
    cv_dir = args.cv_dir or os.path.join(RECORDS_DIR, f"cv_{run_tag}")
    paths = sorted(glob.glob(os.path.join(cv_dir, "fold_*", "suvr_sets_fold*.json")))
    if not paths:
        sys.exit(f"No suvr_sets_fold*.json under {cv_dir}. Run scripts/suvr_sets.py per fold first.")
    folds = [json.load(open(p)) for p in paths]
    fold_ids = [f.get("fold") for f in folds]
    print(f"Aggregating {len(folds)} folds {fold_ids} from {cv_dir}")

    # ── SET 1: scalar metrics, mean ± std across folds ──────────────────────────
    s1_keys = list(folds[0]["set1_pooled_wholebrain"].keys())
    set1 = {k: _ms([f["set1_pooled_wholebrain"][k] for f in folds]) for k in s1_keys}

    # ── SET 2 / SET 3: per-region pearson across folds (regions aligned by name) ─
    def per_region(setkey):
        regions = folds[0][setkey]["regions"]
        mat = np.array([f[setkey]["pearson"] for f in folds], dtype=float)  # (n_folds, 86)
        mean = np.nanmean(mat, axis=0); std = np.nanstd(mat, axis=0)
        df = pd.DataFrame({"R_mean": mean, "R_std": std}, index=regions); df.index.name = "region"
        # fold-level overall: each fold's mean-over-regions, then mean ± std across folds
        overall = _ms([np.nanmean(f[setkey]["pearson"]) for f in folds])
        return df, overall

    set2_df, set2_overall = per_region("set2_region_pooled")
    set3_df, set3_overall = per_region("set3_region_xsub")
    set2_df.to_csv(os.path.join(cv_dir, "region_pooled_SUVR_cv.csv"))
    set3_df.to_csv(os.path.join(cv_dir, "region_crosssubject_SUVR_cv.csv"))

    summary = {
        "mode": args.mode, "n_folds": len(folds), "folds": fold_ids, "space": "SUVR",
        "set1_pooled_wholebrain": {k: {"mean": m, "std": s, "n": n} for k, (m, s, n) in set1.items()},
        "set2_region_pooled_overall":   {"mean": set2_overall[0], "std": set2_overall[1]},
        "set3_region_crosssubject_overall": {"mean": set3_overall[0], "std": set3_overall[1]},
    }
    with open(os.path.join(cv_dir, "suvr_sets_cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── report ──────────────────────────────────────────────────────────────────
    print(f"\nCV — 3 SUVR Pearson R's, mean ± std across {len(folds)} folds ({args.mode})\n" + "=" * 64)
    m, s, _ = set1["pearson"]
    print(f"[1] Pooled voxel-level, whole brain   R = {m:.4f} ± {s:.4f}")
    print(f"[2] Pooled voxel-level, per region    R = {set2_overall[0]:.4f} ± {set2_overall[1]:.4f}  (overall over 86)")
    print(f"[3] Regional cross-subject (86)       R = {set3_overall[0]:.4f} ± {set3_overall[1]:.4f}  (overall over 86)")
    print("\nSET 1 full whole-brain metrics (SUVR):")
    for k, (m, s, n) in set1.items():
        print(f"    {k.upper():<8} {m:.4f} ± {s:.4f}")
    print(f"\nWrote → {cv_dir}/suvr_sets_cv_summary.json")
    print(f"        {cv_dir}/region_pooled_SUVR_cv.csv  (SET 2, 86 regions)")
    print(f"        {cv_dir}/region_crosssubject_SUVR_cv.csv  (SET 3, 86 regions)")


if __name__ == "__main__":
    main()
