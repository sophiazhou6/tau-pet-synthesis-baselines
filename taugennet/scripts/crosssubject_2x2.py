#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""Standalone 2×2 (intensity × scale) cross-subject diagnostic from cached generations.

Computes the four cross-subject Pearson cells WITHOUT running the full evaluate_final.py
job (no model load, no ablation/plasma/figures):

                 regional-mean        voxelwise
    SUVR         M1  (burden)         M2s (burden, voxel)
    normalized   M1n (pattern)        M2  (pattern, voxel)

All metric definitions are imported from evaluate_final so the numbers are identical to
the in-job ones. SUVR cells keep absolute tau burden; normalized [0,1] cells divide it out
(per-subject min-max), isolating spatial pattern.

NOTE: none of these is the paper's regional metric (eqs 16-18, SUVR + plasma grouping).

Usage:
    python scripts/crosssubject_2x2.py --mode atrophy --arch silu
    python scripts/crosssubject_2x2.py --mode combined --arch silu
    python scripts/crosssubject_2x2.py --mode ptau217 --generated-dir results/generated/silu/ptau217
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_final import (
    load_cached_generations,
    compute_regional_overall,
    compute_regional_overall_normalized,
    compute_crosssubject_voxelwise_r,
)


def _build_test_ds(mode, arch):
    """Build the test dataset exactly like crosssubject_r.py (v2 vs combined per mode/arch)."""
    if arch == "relu":
        from src import dataset_v2_relu as _dv2
        from src import dataset_combined_relu as _dc
    else:
        from src import dataset_final as _dv2
        from src import dataset_combined as _dc
    if mode == "combined":
        _, _, test_ds, *_ = _dc.build_dataloaders(mode="combined")
    else:
        _, _, test_ds, *_ = _dv2.build_dataloaders(mode=mode)
    return test_ds


def _roi_mean(pearson_table):
    """Mean over ROIs of a 1-row regional Pearson table (DataFrame)."""
    return float(np.nanmean(pearson_table.to_numpy(dtype=float)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["atrophy", "ptau217", "combined"])
    p.add_argument("--arch", default="silu", choices=["silu", "relu"])
    p.add_argument("--generated-dir", default=None,
                   help="Cached generations dir (default: results/generated/<arch>/<mode>)")
    args = p.parse_args()

    gen_dir = args.generated_dir or os.path.join("results", "generated", args.arch, args.mode)
    print(f"Loading cached generations from: {gen_dir}")

    test_ds = _build_test_ds(args.mode, args.arch)
    real_norm, gen_masked, gen_raw, real_suvr, gen_suvr, _ = \
        load_cached_generations(test_ds, args.mode, gen_dir)
    n = len(gen_raw)
    print(f"Loaded {n} subjects.\n")

    # Shared brain mask → M2 and M2s cover identical voxels (only intensity differs).
    brain_mask = (np.stack(real_norm, axis=0) > 0).any(axis=0)
    gen_suvr_masked = [g * (r > 0) for g, r in zip(gen_suvr, real_norm)]

    # Regional-mean cross-subject Pearson (M1 SUVR, M1n normalized)
    regional_suvr_tables, _, _ = compute_regional_overall(real_suvr, gen_suvr)
    m1  = _roi_mean(regional_suvr_tables["pearson"])
    m1n = _roi_mean(compute_regional_overall_normalized(real_norm, gen_raw))

    # Voxelwise cross-subject r (M2 normalized, M2s SUVR)
    r_norm = compute_crosssubject_voxelwise_r(real_norm, gen_masked,
                                              brain_mask=brain_mask, label=" [normalized]")
    r_suvr = compute_crosssubject_voxelwise_r(real_suvr, gen_suvr_masked,
                                              brain_mask=brain_mask, label=" [SUVR]")
    m2  = float(r_norm[brain_mask].mean())
    m2s = float(r_suvr[brain_mask].mean())

    print("\n" + "=" * 60)
    print(f"  2×2 CROSS-SUBJECT PEARSON — {args.mode} ({args.arch}, N={n})")
    print("  (NOT a paper metric — supplementary diagnostic)")
    print("=" * 60)
    print(f"  {'':<14}{'regional-mean':>16}{'voxelwise':>16}")
    print(f"  {'SUVR':<14}{m1:>16.4f}{m2s:>16.4f}")
    print(f"  {'normalized':<14}{m1n:>16.4f}{m2:>16.4f}")
    print("=" * 60)
    print("  Intensity effect (burden): SUVR vs normalized within a column")
    print("  Scale effect: regional vs voxelwise within a row")


if __name__ == "__main__":
    main()
