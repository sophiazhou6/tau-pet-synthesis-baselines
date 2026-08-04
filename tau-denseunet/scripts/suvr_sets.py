#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Score one fold's cached predictions and emit: the whole-brain per-subject metrics
(metrics.json, for CV aggregation), the three SUVR Pearson-R SETs (JSON + CSVs),
and the SUVR figures — all over a chosen mask.

  --mask-mode dk86        (default) DK86 atlas region (~8.7% of volume)
  --mask-mode wholebrain  whole brain excluding background, from the T1 seg
                          consensus mask (denseunet/masks.py, ~25.5%)

Reuses taugennet/evaluate_final's own metric functions (imported as a module, not
run) on the appropriate mask, so nothing is reimplemented:
  metrics.json  ← compute_wholebrain_metrics (per-subject, normalized [0,1])
  SET 1/2/3     ← compute_pooled_wholebrain_metrics / compute_region_pooled_metrics
                  / compute_region_crosssubject_metrics, in SUVR (un-normalized)

For --mask-mode wholebrain the mask is injected into the dataset's shared mask slot
so the pipeline scores the whole brain instead of DK86 (evaluate_final.py is
DK86-hardcoded and can't be reused directly, so this script is the scorer there).

Usage:
    python scripts/suvr_sets.py --mode atrophy --fold 0
    python scripts/suvr_sets.py --mode atrophy --fold 0 --mask-mode wholebrain
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import (  # noqa: E402
    TAUGENNET_ROOT, REPO_ROOT, GENERATED_DIR, N_FOLDS, build_dataloaders)
from denseunet import figs  # noqa: E402
from denseunet.masks import load_or_build_wholebrain_mask, inject_mask  # noqa: E402

RECORDS_DIR = os.path.join(REPO_ROOT, "results", "records")
FIGURES_DIR = os.path.join(REPO_ROOT, "results", "figures")


def _load_evaluate_final():
    for pth in (TAUGENNET_ROOT, os.path.join(TAUGENNET_ROOT, "scripts")):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    path = os.path.join(TAUGENNET_ROOT, "scripts", "evaluate_final.py")
    spec = importlib.util.spec_from_file_location("evaluate_final_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description="Whole-brain metrics + SUVR SET 1/2/3 R for one fold.")
    p.add_argument("--mode", choices=["atrophy", "ptau217"], required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--mask-mode", choices=["dk86", "wholebrain"], default="dk86")
    p.add_argument("--generated-dir", default=None)
    args = p.parse_args()

    run_tag = args.mode if args.mask_mode == "dk86" else f"{args.mode}_wholebrain"
    fold_sub = "" if args.fold is None else f"fold_{args.fold}"
    gen_dir = args.generated_dir or os.path.join(GENERATED_DIR, run_tag, fold_sub)
    out_dir = (os.path.join(RECORDS_DIR, f"cv_{run_tag}", fold_sub) if args.fold is not None
               else os.path.join(RECORDS_DIR, f"single_{run_tag}"))
    fig_dir = os.path.join(FIGURES_DIR, run_tag, fold_sub, "suvr")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(gen_dir) or not any(f.endswith(".npy") for f in os.listdir(gen_dir)):
        sys.exit(f"No cached predictions in {gen_dir}.")

    ef = _load_evaluate_final()
    from src.dataset_final import unnormalize  # noqa: E402

    fold_kwargs = {} if args.fold is None else {"fold_idx": args.fold, "n_folds": args.n_folds}
    train_ds, val_ds, test_ds, *_ = build_dataloaders(mode=args.mode, use_dk_mask=True, **fold_kwargs)
    if args.mask_mode == "wholebrain":
        inject_mask(load_or_build_wholebrain_mask(args.mode), train_ds, val_ds, test_ds)

    mask = ef._dk86_mask(test_ds)          # (H,W,D) bool — DK86 atlas OR injected whole-brain
    bbox = ef._mask_bbox(mask)
    label_vol = ef._dk86_label_volume(test_ds)   # atlas labels (SET 2/3 stay region-defined)

    real_norm, gen_masked, real_suvr, gen_suvr = [], [], [], []
    for i in range(len(test_ds)):
        real = test_ds[i][0].squeeze().numpy()          # normalized, masked to `mask`
        gen = np.load(os.path.join(gen_dir, f"subject_{i:03d}.npy")).astype(np.float32)
        pmin, pmax = test_ds.get_pet_norms(i)
        real_norm.append(real)
        gen_masked.append(gen * mask)
        real_suvr.append(unnormalize(real, pmin, pmax))
        gen_suvr.append(unnormalize(gen, pmin, pmax) * mask)

    # ── whole-brain per-subject metrics (normalized [0,1]) → metrics.json ────────
    pears, nrmses, ssims, mses, maes = ef.compute_wholebrain_metrics(
        real_norm, gen_masked, brain_mask=mask, bbox=bbox)
    metrics = {"pearson": float(np.nanmean(pears)), "nrmse": float(np.nanmean(nrmses)),
               "ssim": float(np.nanmean(ssims)), "mse": float(np.nanmean(mses)),
               "mae": float(np.nanmean(maes)), "n": len(test_ds), "fold": args.fold,
               "mask_mode": args.mask_mode}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── SUVR SET 1/2/3 (reuse evaluate_final's SET functions on SUVR volumes) ────
    set1 = ef.compute_pooled_wholebrain_metrics(real_suvr, gen_suvr, mask)
    set2 = ef.compute_region_pooled_metrics(real_suvr, gen_suvr, label_vol)
    set3, _Rr, _Rg = ef.compute_region_crosssubject_metrics(real_suvr, gen_suvr, label_vol)
    set1.to_csv(os.path.join(out_dir, "pooled_wholebrain_SUVR.csv"))
    set2.to_csv(os.path.join(out_dir, "region_pooled_SUVR.csv"))
    set3.to_csv(os.path.join(out_dir, "region_crosssubject_SUVR.csv"))

    payload = {
        "mode": args.mode, "fold": args.fold, "mask_mode": args.mask_mode,
        "n_test": len(test_ds), "space": "SUVR",
        "wholebrain_metrics_normalized": metrics,
        "set1_pooled_wholebrain": {k: float(set1[k].iloc[0]) for k in set1.columns},
        "set2_region_pooled": {"regions": list(set2.index),
                               **{k: [float(x) for x in set2[k].values] for k in set2.columns}},
        "set3_region_xsub": {"regions": list(set3.index),
                             **{k: [float(x) for x in set3[k].values] for k in set3.columns}},
    }
    with open(os.path.join(out_dir, f"suvr_sets_fold{args.fold}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # ── figures (fixed layout — no clipping) ────────────────────────────────────
    r_all = np.concatenate([r[mask].ravel() for r in real_suvr])
    g_all = np.concatenate([g[mask].ravel() for g in gen_suvr])
    figs.plot_suvr_figures(r_all, g_all, float(set1["pearson"].iloc[0]), set2, set3, fig_dir,
                           tag=f"({args.mask_mode})")

    print(f"\n[{run_tag}] fold {args.fold}  (N={len(test_ds)}, mask={args.mask_mode})")
    print(f"  whole-brain  Pearson={metrics['pearson']:.4f}  SSIM={metrics['ssim']:.4f}  "
          f"MSE={metrics['mse']:.5f}  (normalized)")
    print(f"  SET1 pooled whole R = {payload['set1_pooled_wholebrain']['pearson']:.4f}")
    print(f"  SET2 per-region R   = mean {np.nanmean(set2['pearson']):.4f}")
    print(f"  SET3 cross-subj R   = mean {np.nanmean(set3['pearson']):.4f}")
    print(f"  → records {out_dir}\n  → figures {fig_dir}")


if __name__ == "__main__":
    main()
