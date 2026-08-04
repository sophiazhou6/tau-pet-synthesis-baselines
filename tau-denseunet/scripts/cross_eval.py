#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Cross-mask evaluation: score the WHOLE-BRAIN-trained model over the DK86 region.

The whole-brain model's cached predictions live in whole-brain-normalized space, so
they are un-normalized to SUVR with each subject's whole-brain min/max, then scored
over the DK86 atlas region. SUVR is mask-independent, so the result is directly
comparable to the DK86-trained model's DK86 SUVR numbers (results/records/cv_atrophy).

Per fold → results/records/cv_atrophy_wb_on_dk86/fold_i/, then mean ± std across folds.

Usage:
    python scripts/cross_eval.py --mode atrophy
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import TAUGENNET_ROOT, REPO_ROOT, GENERATED_DIR, N_FOLDS, build_dataloaders  # noqa: E402
from denseunet.masks import load_or_build_wholebrain_mask, inject_mask  # noqa: E402

RECORDS_DIR = os.path.join(REPO_ROOT, "results", "records")


def _load_ef():
    for pth in (TAUGENNET_ROOT, os.path.join(TAUGENNET_ROOT, "scripts")):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    path = os.path.join(TAUGENNET_ROOT, "scripts", "evaluate_final.py")
    spec = importlib.util.spec_from_file_location("evaluate_final_mod", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _ms(vals):
    a = np.asarray(vals, float); a = a[np.isfinite(a)]
    return float(a.mean()), (float(a.std()) if a.size > 1 else 0.0)


def render_glass_brain_on_dk86(mode, ef, rank_by="ssim"):
    """Glass brains of the whole-brain model's ENSEMBLE predictions, displayed over
    the DK86 region (so it visually matches the DK86 model's glass brain). Un-normalize
    with whole-brain norms → SUVR, then restrict to the DK86 atlas. `rank_by` selects
    the Best/Median/Worst subjects (ssim | mse | mae | nrmse | pearson)."""
    from glass_brain_final import run_mode  # noqa: E402
    from src.dataset_final import unnormalize  # noqa: E402

    ens_dir = os.path.join(GENERATED_DIR, f"{mode}_wholebrain", "ensemble")
    _, _, test_ds, *_ = build_dataloaders(mode=mode, use_dk_mask=True)
    inject_mask(load_or_build_wholebrain_mask(mode), test_ds)   # whole-brain norms
    dk = ef._atlas_dk86_binary()

    real_suvr, gb_gen_suvr, real_norm, gen_norm = [], [], [], []
    for i in range(len(test_ds)):
        real = test_ds[i][0].squeeze().numpy()
        gen = np.load(os.path.join(ens_dir, f"subject_{i:03d}.npy")).astype(np.float32)
        wmin, wmax = test_ds.get_pet_norms(i)
        real_suvr.append(unnormalize(real, wmin, wmax) * dk)   # SUVR, DK86 only
        gb_gen_suvr.append(unnormalize(gen, wmin, wmax) * dk)
        real_norm.append(real * dk); gen_norm.append(gen * dk)

    gb_dir = os.path.join(REPO_ROOT, "results", "figures", f"{mode}_wb_on_dk86", "glass_brain")
    run_mode(mode, "silu", gb_dir, use_dk_mask=True,
             data=(real_suvr, gb_gen_suvr, real_norm, gen_norm, test_ds), rank_by=rank_by)
    print(f"[wb-model → DK86 glass brain, rank_by={rank_by}] → {gb_dir}/{rank_by}/")


def main():
    p = argparse.ArgumentParser(description="Score the whole-brain model over DK86 (SUVR).")
    p.add_argument("--mode", choices=["atrophy", "ptau217"], required=True)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--glass-brain", action="store_true",
                   help="Also render whole-brain-model glass brains displayed over DK86.")
    args = p.parse_args()

    ef = _load_ef()
    from src.dataset_final import unnormalize  # noqa: E402

    if args.glass_brain:
        render_glass_brain_on_dk86(args.mode, ef)

    per_fold = []
    for fold in range(args.n_folds):
        gen_dir = os.path.join(GENERATED_DIR, f"{args.mode}_wholebrain", f"fold_{fold}")
        if not os.path.isdir(gen_dir):
            print(f"fold {fold}: no whole-brain cache, skipping"); continue

        # whole-brain-injected test_ds → whole-brain per-subject norms (how gen was normalized)
        _, _, test_ds, *_ = build_dataloaders(mode=args.mode, fold_idx=fold,
                                              n_folds=args.n_folds, use_dk_mask=True)
        inject_mask(load_or_build_wholebrain_mask(args.mode), test_ds)
        dk = ef._atlas_dk86_binary()            # DK86 region (H,W,D bool) — the SCORING mask
        label_vol = ef._dk86_label_volume(test_ds)
        bbox = ef._mask_bbox(dk)

        real_suvr, gen_suvr, real_dk_norm, gen_dk_masked = [], [], [], []
        for i in range(len(test_ds)):
            real = test_ds[i][0].squeeze().numpy()      # whole-brain-normalized real
            gen = np.load(os.path.join(gen_dir, f"subject_{i:03d}.npy")).astype(np.float32)
            wmin, wmax = test_ds.get_pet_norms(i)       # whole-brain norms
            rs = unnormalize(real, wmin, wmax)
            gs = unnormalize(gen, wmin, wmax)
            real_suvr.append(rs); gen_suvr.append(gs * dk)
            real_dk_norm.append(real); gen_dk_masked.append(gen * dk)

        # whole-brain per-subject metrics over DK86 (normalized-in-wb-space, for reference)
        pe, nr, ss, ms, ma = ef.compute_wholebrain_metrics(real_dk_norm, gen_dk_masked,
                                                           brain_mask=dk, bbox=bbox)
        # SUVR SET 1 over DK86 — the comparable-to-cv_atrophy numbers
        set1 = ef.compute_pooled_wholebrain_metrics(real_suvr, gen_suvr, dk)

        out = os.path.join(RECORDS_DIR, f"cv_{args.mode}_wb_on_dk86", f"fold_{fold}")
        os.makedirs(out, exist_ok=True)
        rec = {"fold": fold, "n": len(test_ds),
               "suvr_set1": {k: float(set1[k].iloc[0]) for k in set1.columns},
               "normalized_wb_space": {"pearson": float(np.nanmean(pe)),
                                       "mse": float(np.nanmean(ms)), "mae": float(np.nanmean(ma))}}
        json.dump(rec, open(os.path.join(out, "metrics.json"), "w"), indent=2)
        per_fold.append(rec)
        print(f"fold {fold}: SUVR SET1 R={rec['suvr_set1']['pearson']:.4f} "
              f"MSE={rec['suvr_set1']['mse']:.4f} MAE={rec['suvr_set1']['mae']:.4f}")

    # aggregate SUVR SET1 across folds
    agg = {k: _ms([f["suvr_set1"][k] for f in per_fold])
           for k in ["pearson", "nrmse", "ssim", "mse", "mae"]}
    summary = {"mode": args.mode, "n_folds": len(per_fold), "space": "SUVR over DK86",
               "suvr_set1": {k: {"mean": m, "std": s} for k, (m, s) in agg.items()}}
    cv_dir = os.path.join(RECORDS_DIR, f"cv_{args.mode}_wb_on_dk86")
    json.dump(summary, open(os.path.join(cv_dir, "summary.json"), "w"), indent=2)

    print(f"\n=== Whole-brain model scored over DK86 (SUVR, {len(per_fold)}-fold mean ± std) ===")
    for k in ["pearson", "nrmse", "ssim", "mse", "mae"]:
        m, s = agg[k]; print(f"  {k.upper():8} {m:.4f} ± {s:.4f}")
    print(f"→ {cv_dir}/summary.json")


if __name__ == "__main__":
    main()
