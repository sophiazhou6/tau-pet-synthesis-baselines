#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
eval_mentor_metrics.py — the mentor's 3 canonical Pearson r's, ALL in SUVR, DK86-masked.

  1. Pooled voxel R   : concat every DK86 voxel SUVR over all subjects (pred vs real) -> 1 r
  2. Regional R (x86) : per region, mean voxel SUVR per subject -> N-vector; r vs real; x86 -> mean
  3. Subject R        : whole-brain (DK86) mean SUVR per subject -> N-vector; r vs real -> 1 r

SUVR = unnormalize with the dataset's masked get_pet_norms (NEVER raw-NIfTI max). Evaluate cached
generations with the SAME split they were made under (legacy for all existing gens).

Usage (single model):
  eval_mentor_metrics.py --kind spatial --cond-mode none --fold 0 \
      --generated-dir results/generated/cond_compare_maskfix/spatialmap_none/fold_0
Usage (all four canonical models):
  eval_mentor_metrics.py --all
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import pearsonr
import evaluate_final as ev


def _build_test_ds(kind, mode, cond_mode, fold, n_folds, use_mentor_split):
    fk = {} if fold is None else {"fold_idx": fold, "n_folds": n_folds}
    if kind == "combined":
        from src import dataset_combined as d
        _, _, test_ds, *_ = d.build_dataloaders(mode="combined", use_dk_mask=True,
                                                 use_mentor_split=use_mentor_split, **fk)
    elif kind == "spatial":
        from src import dataset_spatial as d
        _, _, test_ds, *_ = d.build_dataloaders(use_dk_mask=True, cond_mode=cond_mode,
                                                 use_mentor_split=use_mentor_split, **fk)
    else:  # final
        from src import dataset_final as d
        _, _, test_ds, *_ = d.build_dataloaders(mode=mode, use_dk_mask=True,
                                                 use_mentor_split=use_mentor_split, **fk)
    return test_ds


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 2 or a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return float(pearsonr(a, b)[0])


def eval_model(name, kind, mode, cond_mode, fold, generated_dir,
               n_folds=5, use_mentor_split=False):
    test_ds = _build_test_ds(kind, mode, cond_mode, fold, n_folds, use_mentor_split)
    cm = mode if kind != "spatial" else cond_mode
    real_norm, gen_masked, gen_raw, real_suvr, gen_suvr, _ = \
        ev.load_cached_generations(test_ds, cm, generated_dir)

    dk = ev._dk86_mask(test_ds)                       # (H,W,D) bool DK86
    lab = ev._dk86_label_volume(test_ds)              # (H,W,D) int 0..86
    R = np.stack([r * dk for r in real_suvr])         # (N,H,W,D) SUVR, DK86
    G = np.stack([g * dk for g in gen_suvr])
    N = R.shape[0]
    m = dk

    # 1. Pooled voxel R (all DK86 voxels, all subjects concatenated)
    pooled = _pearson(R[:, m].ravel(), G[:, m].ravel())

    # 2. Regional R x86 (per region: N subject-means -> r vs real)
    reg_r = []
    for k in range(1, 87):
        rm = (lab == k)
        if not rm.any():
            reg_r.append(np.nan); continue
        rv = np.array([R[i][rm].mean() for i in range(N)])
        gv = np.array([G[i][rm].mean() for i in range(N)])
        reg_r.append(_pearson(rv, gv))
    reg_r = np.array(reg_r)

    # 3. Subject R (whole-brain DK86 mean per subject -> r vs real)
    r_sub = np.array([R[i][m].mean() for i in range(N)])
    g_sub = np.array([G[i][m].mean() for i in range(N)])
    subject = _pearson(r_sub, g_sub)

    return dict(name=name, N=N, pooled=pooled,
                regional_mean=float(np.nanmean(reg_r)),
                regional_median=float(np.nanmedian(reg_r)),
                subject=subject, reg_r=reg_r)


CANONICAL = [
    dict(name="silu_combined (full brain)", kind="combined", mode="combined", cond_mode=None,
         fold=None, generated_dir="results/generated/silu_v3/combined"),
    dict(name="cond_a_mlp (atrophy MLP)",   kind="final",    mode="atrophy",  cond_mode=None,
         fold=0,    generated_dir="results/generated/dk86_regen/cond_a_mlp/fold_0"),
    dict(name="cond_b (spatial map)",       kind="spatial",  mode="atrophy",  cond_mode="none",
         fold=0,    generated_dir="results/generated/cond_compare_maskfix/spatialmap_none/fold_0"),
    dict(name="cond_c (spatial map+ptau)",  kind="spatial",  mode="atrophy",  cond_mode="ptau217",
         fold=0,    generated_dir="results/generated/dk86_regen/cond_c_ptau/fold_0"),
]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--kind", choices=["combined", "spatial", "final"])
    p.add_argument("--mode", default="atrophy")
    p.add_argument("--cond-mode", default="none")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--generated-dir")
    p.add_argument("--name", default="model")
    args = p.parse_args()

    models = CANONICAL if args.all else [dict(
        name=args.name, kind=args.kind, mode=args.mode, cond_mode=args.cond_mode,
        fold=args.fold, generated_dir=args.generated_dir)]

    rows = [eval_model(**{k: v for k, v in mm.items()}) for mm in models]
    print("\n=== MENTOR METRICS — SUVR, DK86 (Pearson r) ===")
    print(f"{'model':32s} {'N':>3s} {'PooledVoxelR':>13s} {'RegionalR(86 mean)':>19s} {'RegionalR(med)':>15s} {'SubjectR':>9s}")
    for r in rows:
        print(f"{r['name']:32s} {r['N']:3d} {r['pooled']:13.3f} {r['regional_mean']:19.3f} "
              f"{r['regional_median']:15.3f} {r['subject']:9.3f}")
