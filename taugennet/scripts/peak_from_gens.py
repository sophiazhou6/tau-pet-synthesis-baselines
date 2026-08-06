#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""peak_from_gens.py — compute SET3 / WITHIN / PEAK directly from cached .npy generations.

Use when a run did NOT write regional_means_*.csv (e.g. evaluate_cfg.py). Otherwise prefer
scripts/localization_metrics.py, which reads those CSVs.

    peak_from_gens.py --dataset combined --mode combined \
        results/generated/cfg_sweep/combined_w5 results/generated/cfg_sweep/combined_w15

Metrics (all in SUVR, DK86-masked):
  SET3    per region, ACROSS subjects   -> inter-subject burden level
  WITHIN  per subject, ACROSS regions   -> intra-brain gradient
  PEAK@K  per subject, over that subject's K hottest REAL regions -> localization  <-- select on this
  NRMSE / contrast are printed too: PEAK is per-subject scale-invariant and therefore CANNOT
  detect over-saturation on its own. Always read them together.
"""
import argparse, os, sys
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
ef = importlib.import_module("evaluate_final")


def _sr(a, b):
    return np.nan if a.std() < 1e-9 or b.std() < 1e-9 else pearsonr(a, b)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gen_dirs", nargs="+")
    p.add_argument("--dataset", choices=["final", "combined", "spatial"], required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--cond-mode", default="atrophy")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--use-mentor-split", action=argparse.BooleanOptionalAction, default=False,
                   help="MUST match the split the generations were produced under.")
    p.add_argument("--top-k", type=int, default=10)
    a = p.parse_args()

    fk = {} if a.fold is None else dict(fold_idx=a.fold, n_folds=a.n_folds)
    if a.dataset == "combined":
        from src import dataset_combined as D
        ts = D.build_dataloaders(mode=a.mode, use_dk_mask=True,
                                 use_mentor_split=a.use_mentor_split, **fk)[2]
    elif a.dataset == "spatial":
        from src import dataset_spatial as D
        ts = D.build_dataloaders(use_dk_mask=True, cond_mode=a.cond_mode,
                                 use_mentor_split=a.use_mentor_split, **fk)[2]
    else:
        from src import dataset_final as D
        ts = D.build_dataloaders(mode=a.mode, use_dk_mask=True,
                                 use_mentor_split=a.use_mentor_split, **fk)[2]

    lv = ef._dk86_label_volume(ts)
    bm = ef._dk86_mask(ts)
    bm = ef._atlas_dk86_binary() if bm is None else bm
    labels = [r for r in range(1, 87) if (lv == r).sum() > 0]

    def regional(vols):
        return np.array([[v[lv == r].mean() for r in labels] for v in vols])

    load_mode = a.mode if a.dataset != "spatial" else ("atrophy" if a.cond_mode != "ptau217" else a.mode)
    # SET1 pooled voxel whole-brain | SET2 pooled voxel per-region (mean over 86) | SET3 regional cross-subject
    print(f"{'gen_dir':40s} {'SET1':>6s} {'SET2':>6s} {'SET3':>6s} | "
          f"{'WITHIN':>7s} {f'PEAK@{a.top_k}':>8s} | {'NRMSE':>7s} {'contr':>6s}")
    for gd in a.gen_dirs:
        # Skip a missing/empty dir rather than killing the whole table — one failed guidance
        # value should not take out the summary (this runs inside the notify/digest job).
        import glob as _glob
        if not os.path.isdir(gd) or not _glob.glob(os.path.join(gd, "subject_*.npy")):
            print(f"{os.path.basename(gd.rstrip('/')):46s} {'--- no generations ---':>44s}", flush=True)
            continue
        try:
            _, _, _, rs, gs, _ = ef.load_cached_generations(ts, load_mode, gd)
        except Exception as e:
            print(f"{os.path.basename(gd.rstrip('/')):46s}  [load failed: {type(e).__name__}]", flush=True)
            continue
        R, G = regional(rs), regional(gs)
        # SET1: all in-brain voxels of all subjects pooled -> one real vs gen vector -> single R
        r_all = np.concatenate([r[bm].ravel() for r in rs]).astype(np.float64)
        g_all = np.concatenate([g[bm].ravel() for g in gs]).astype(np.float64)
        set1  = _sr(r_all, g_all)
        # SET2: same pooling but restricted to each of the 86 regions -> mean of per-region R
        set2  = np.nanmean([_sr(np.concatenate([r[lv == lab].ravel() for r in rs]),
                                np.concatenate([g[lv == lab].ravel() for g in gs]))
                            for lab in labels])
        set3   = np.nanmean([_sr(R[:, j], G[:, j]) for j in range(R.shape[1])])
        within = np.nanmean([_sr(R[i, :], G[i, :]) for i in range(R.shape[0])])
        peak   = np.nanmean([_sr(R[i, np.argsort(R[i, :])[-a.top_k:]],
                                 G[i, np.argsort(R[i, :])[-a.top_k:]]) for i in range(R.shape[0])])
        nrmse    = np.mean([np.sqrt(((r[bm]-g[bm])**2).mean()) / (r[bm].max()-r[bm].min())
                            for r, g in zip(rs, gs)])
        contrast = np.mean([g[bm].std() for g in gs]) / np.mean([r[bm].std() for r in rs])
        print(f"{os.path.basename(gd.rstrip('/')):40s} {set1:6.3f} {set2:6.3f} {set3:6.3f} | "
              f"{within:7.3f} {peak:8.3f} | {nrmse:7.3f} {contrast:6.2f}", flush=True)

    print("\nSET1/2/3 = your mandated Pearsons (SET3 inter-subject). WITHIN/PEAK = intra-subject localization.")
    print("all SUVR, DK86-masked. NRMSE/contrast guard against over-smoothing (PEAK is scale-invariant).")


if __name__ == "__main__":
    main()
