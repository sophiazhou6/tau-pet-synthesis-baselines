#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Two per-fold products for the Dense-U-Net baseline:

 1. PEAK@K localization metric (+ WITHIN, SET3), computed with taugennet's own
    scripts/localization_metrics.metrics() so it matches exactly. PEAK@K = for each
    subject, take their top-K hottest REAL DK regions, then Pearson real-vs-gen across
    just those K regions; averaged over subjects. It is a WITHIN-subject correlation,
    so it is invariant to the per-subject min-max normalization.

    The DK86 folds already have regional_means_{real,gen}_*.csv (from evaluate_final).
    The whole-brain folds don't (evaluate_final is DK86-only), so we build them here
    with evaluate_final's _regional_means_86 over the 86 DK parcels.

 2. A pooled SET-1 scatter per fold (real vs predicted SUVR, all in-mask voxels of all
    test subjects pooled) → results/figures/<run_tag>/suvr/folds/pooled_scatter_fold<i>.png

Usage:
    python scripts/peak_and_folds.py --mode atrophy --top-k 10
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import TAUGENNET_ROOT, REPO_ROOT, GENERATED_DIR, N_FOLDS, build_dataloaders  # noqa: E402
from denseunet.masks import load_or_build_wholebrain_mask, inject_mask  # noqa: E402
from denseunet import figs  # noqa: E402

RECORDS_DIR = os.path.join(REPO_ROOT, "results", "records")
FIGURES_DIR = os.path.join(REPO_ROOT, "results", "figures")


def _load(name):
    for pth in (TAUGENNET_ROOT, os.path.join(TAUGENNET_ROOT, "scripts")):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    path = os.path.join(TAUGENNET_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_mod", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _ms(v):
    a = np.asarray(v, float); a = a[np.isfinite(a)]
    return float(a.mean()), (float(a.std()) if a.size > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["atrophy", "ptau217"], default="atrophy")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = p.parse_args()

    ef = _load("evaluate_final")
    loc = _load("localization_metrics")
    from src.dataset_final import unnormalize  # noqa: E402
    from scipy.stats import pearsonr  # noqa: E402

    summary = {}
    for mask_mode in ("dk86", "wholebrain"):
        run_tag = args.mode if mask_mode == "dk86" else f"{args.mode}_wholebrain"
        rows = []
        for fold in range(args.n_folds):
            gen_dir = os.path.join(GENERATED_DIR, run_tag, f"fold_{fold}")
            rec_dir = os.path.join(RECORDS_DIR, f"cv_{run_tag}", f"fold_{fold}")
            os.makedirs(rec_dir, exist_ok=True)

            _, _, ds, *_ = build_dataloaders(mode=args.mode, fold_idx=fold,
                                             n_folds=args.n_folds, use_dk_mask=True)
            if mask_mode == "wholebrain":
                inject_mask(load_or_build_wholebrain_mask(args.mode), ds)
            mask = ef._dk86_mask(ds)                    # this run's mask
            label_vol = ef._dk86_label_volume(ds)       # 86 DK parcels (both variants)

            real_n, gen_m, r_pool, g_pool = [], [], [], []
            for i in range(len(ds)):
                real = ds[i][0].squeeze().numpy()
                gen = np.load(os.path.join(gen_dir, f"subject_{i:03d}.npy")).astype(np.float32)
                pmin, pmax = ds.get_pet_norms(i)
                real_n.append(real); gen_m.append(gen * mask)
                rs = unnormalize(real, pmin, pmax); gs = unnormalize(gen, pmin, pmax)
                r_pool.append(rs[mask]); g_pool.append(gs[mask])

            # ── regional means over the 86 DK parcels (write CSVs if absent) ──────
            Rreal = ef._regional_means_86(real_n, label_vol)
            Rgen = ef._regional_means_86(gen_m, label_vol)
            idx = [f"subject_{i:03d}" for i in range(Rreal.shape[0])]
            names = list(ef.REGION_NAMES)
            for arr, nm in ((Rreal, "real"), (Rgen, "gen")):
                fp = os.path.join(rec_dir, f"regional_means_{nm}_{args.mode}.csv")
                if not os.path.exists(fp):
                    pd.DataFrame(arr, index=idx, columns=names).to_csv(fp)

            # ── PEAK@K / WITHIN / SET3 via taugennet's own function ──────────────
            set3, within, peak, ns, nr = loc.metrics(Rreal, Rgen, top_k=args.top_k)
            rows.append({"fold": fold, "SET3": set3, "WITHIN": within, "PEAK": peak})
            print(f"[{run_tag}] fold {fold}: PEAK@{args.top_k}={peak:.4f} "
                  f"WITHIN={within:.4f} SET3={set3:.4f}")

            # ── per-fold pooled SET-1 scatter (SUVR) ─────────────────────────────
            r_all = np.concatenate(r_pool); g_all = np.concatenate(g_pool)
            R1 = float(pearsonr(r_all, g_all)[0])
            out = os.path.join(FIGURES_DIR, run_tag, "suvr", "folds",
                               f"pooled_scatter_fold{fold}.png")
            figs.plot_pooled_scatter(r_all, g_all, R1, out, tag=f"({run_tag}, fold {fold})")

        agg = {k: _ms([r[k] for r in rows]) for k in ("PEAK", "WITHIN", "SET3")}
        summary[run_tag] = {"per_fold": rows, "top_k": args.top_k,
                            **{k: {"mean": m, "std": s} for k, (m, s) in agg.items()}}
        print(f"\n=== {run_tag} ({args.n_folds}-fold mean ± std) ===")
        for k in ("PEAK", "WITHIN", "SET3"):
            lbl = f"PEAK@{args.top_k}" if k == "PEAK" else k
            print(f"  {lbl:10} {agg[k][0]:.4f} ± {agg[k][1]:.4f}")
        print()

    out_json = os.path.join(RECORDS_DIR, f"localization_peak{args.top_k}_{args.mode}.json")
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"→ {out_json}")


if __name__ == "__main__":
    main()
