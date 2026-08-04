#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""suvr_eval_figures.py — corrected-SUVR SET1/2/3 figures (+ SET2-vs-SET3) and glass brain
for a cached model. Evaluation ALWAYS uses the legacy split (matches all existing gens) and
true SUVR (via the datasets' get_pet_norms — no raw-max inflation).

Emits into --out:
  SET1_pooled_wholebrain_scatter_SUVR.png    pooled in-brain voxels, real vs gen (hexbin)
  SET2_region_pooled_bar_SUVR.png            86-region pooled-voxel Pearson R
  SET3_region_crosssubject_bar_SUVR.png      86-region cross-subject Pearson R
  SET2_vs_SET3_scatter_SUVR.png              per-region pooled vs cross-subject
  set1_SUVR.csv / set2_SUVR.csv / set3_SUVR.csv
  glass_brain/                               MIP glass/slice/scatter (glass_brain_final.run_mode)

Usage:
  suvr_eval_figures.py --dataset final    --mode atrophy  --gen-dir <dir> --fold 0 --out <dir>
  suvr_eval_figures.py --dataset spatial  --mode atrophy  --cond-mode none    --gen-dir <dir> --fold 0 --out <dir>
  suvr_eval_figures.py --dataset combined --mode combined --gen-dir <dir>            --out <dir>
"""
import argparse, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import pearsonr
import evaluate_final as ef


def _build_test_ds(dataset, mode, cond_mode, fold, n_folds):
    fk = {} if fold is None else dict(fold_idx=fold, n_folds=n_folds)
    if dataset == "spatial":
        from src import dataset_spatial as D
        return D.build_dataloaders(use_dk_mask=True, cond_mode=cond_mode,
                                   use_mentor_split=False, **fk)[2]
    if dataset == "combined":
        from src import dataset_combined as D
        return D.build_dataloaders(mode="combined", use_dk_mask=True,
                                   use_mentor_split=False, **fk)[2]
    from src import dataset_final as D
    return D.build_dataloaders(mode, use_dk_mask=True, use_mentor_split=False, **fk)[2]


def _bar(tbl, title, path):
    s = tbl["pearson"].dropna().sort_values()                 # highest ends at top
    cmap = plt.get_cmap("RdYlGn"); norm = Normalize(vmin=-0.2, vmax=0.7)
    fig, ax = plt.subplots(figsize=(9, 16))
    ax.barh(range(len(s)), s.values, color=cmap(norm(s.values)))
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=5)
    m = float(s.mean())
    ax.axvline(m, color="navy", ls="--", lw=1.2, label=f"mean R = {m:.3f}")
    ax.set_xlabel("Pearson R (SUVR)"); ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right"); ax.margins(y=0.005)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["final", "combined", "spatial"], required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--cond-mode", default="atrophy")
    p.add_argument("--gen-dir", required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--arch", default="silu")
    p.add_argument("--out", required=True)
    p.add_argument("--no-glass-brain", action="store_true")
    p.add_argument("--rank-by", choices=["pearson", "ssim", "mae"], default="pearson",
                   help="rank glass/slice/scatter B best/median/worst by this per-subject metric "
                        "(mae: lower is better -> Best = lowest error). Saved under glass_brain/<metric>/.")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ts = _build_test_ds(a.dataset, a.mode, a.cond_mode, a.fold, a.n_folds)
    load_mode = a.mode if a.dataset != "spatial" else ("atrophy" if a.cond_mode != "ptau217" else a.mode)
    rn, gm, gr, rs, gs, pl = ef.load_cached_generations(ts, load_mode, a.gen_dir)
    assert hasattr(ts, "get_pet_norms"), "dataset lacks get_pet_norms -> SUVR would be raw-inflated"
    bm = ef._dk86_mask(ts); bm = ef._atlas_dk86_binary() if bm is None else bm
    lv = ef._dk86_label_volume(ts)
    N = len(rs)

    # ---- SET 1: pooled whole-brain ----
    r_all = np.concatenate([v[bm].ravel() for v in rs]).astype(np.float64)
    g_all = np.concatenate([v[bm].ravel() for v in gs]).astype(np.float64)
    r1 = float(pearsonr(r_all, g_all)[0])
    import pandas as pd
    pd.DataFrame([{"pearson": r1, "n_voxels": r_all.size, "n_subj": N}]).to_csv(os.path.join(a.out, "set1_SUVR.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    hb = ax.hexbin(r_all, g_all, gridsize=80, bins="log", cmap="viridis", mincnt=1)
    hi = float(np.percentile(np.concatenate([r_all, g_all]), 99.5))
    ax.plot([0, hi], [0, hi], "r--", lw=1.2, label="identity"); ax.legend(loc="upper left")
    ax.set_xlabel("Real tau SUVR"); ax.set_ylabel("Predicted tau SUVR")
    ax.set_title(f"SET 1 — Pooled voxel-level, whole brain (SUVR)\n"
                 f"Pearson R = {r1:.3f}   (N={N} subjects, all DK86 voxels pooled)")
    fig.colorbar(hb, ax=ax, label="log10(voxel count)")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "SET1_pooled_wholebrain_scatter_SUVR.png"), dpi=130); plt.close(fig)

    # ---- SET 2 / SET 3 ----
    set2 = ef.compute_region_pooled_metrics(rs, gs, lv); set2.to_csv(os.path.join(a.out, "set2_SUVR.csv"))
    o3 = ef.compute_region_crosssubject_metrics(rs, gs, lv)
    set3 = o3[0] if isinstance(o3, tuple) else o3; set3.to_csv(os.path.join(a.out, "set3_SUVR.csv"))
    _bar(set2, "SET 2 — Pooled voxel-level Pearson R per DK region (SUVR)\n(R over pooled voxels within each region)",
         os.path.join(a.out, "SET2_region_pooled_bar_SUVR.png"))
    _bar(set3, "SET 3 — Regional cross-subject Pearson R per DK region (SUVR)\n(region mean per subject, R across subjects)",
         os.path.join(a.out, "SET3_region_crosssubject_bar_SUVR.png"))

    # ---- SET2 vs SET3 ----
    j = set2[["pearson"]].join(set3[["pearson"]], lsuffix="_2", rsuffix="_3").dropna()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(j["pearson_2"], j["pearson_3"], s=28, alpha=0.8)
    lo = float(min(j.min().min(), 0) - 0.05); hi = float(j.max().max() + 0.05)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("SET 2: pooled-voxel R (SUVR)"); ax.set_ylabel("SET 3: cross-subject R (SUVR)")
    ax.set_title("Per-region: pooled-voxel vs cross-subject R (SUVR)")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "SET2_vs_SET3_scatter_SUVR.png"), dpi=130); plt.close(fig)

    print(f"[{a.out}] N={N}  SET1={r1:.3f}  SET2_mean={set2['pearson'].mean():.3f}  SET3_mean={set3['pearson'].mean():.3f}")

    # ---- glass brain (true SUVR) ----
    if not a.no_glass_brain:
        try:
            from glass_brain_final import run_mode as gb
            gb_dir = os.path.join(a.out, "glass_brain")
            os.makedirs(gb_dir, exist_ok=True)          # run_mode savefig()s here; must exist
            gb(load_mode, a.arch, gb_dir,
               use_dk_mask=True, data=(rs, gs, rn, gm, ts), rank_by=a.rank_by)
            print("  glass brain ->", os.path.join(a.out, "glass_brain"))
        except Exception as e:
            print("  glass brain skipped:", e)


if __name__ == "__main__":
    main()
