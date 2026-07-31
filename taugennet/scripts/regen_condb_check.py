#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""Evaluate (mentor metrics) AND render glass brains for a cond_b generation dir, from the
SAME volumes + SAME dataset build, so the reported Regional R and the glass brains are
guaranteed self-consistent. Spatial dataset, legacy fold-0, cond=none (cond_b's training split)."""
import warnings; warnings.filterwarnings("ignore")
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import pearsonr
import evaluate_final as ev
from glass_brain_final import run_mode
from src import dataset_spatial as ds

p = argparse.ArgumentParser()
p.add_argument("--generated-dir", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--tag", default="regen")
a = p.parse_args()

_, _, test_ds, *_ = ds.build_dataloaders(use_dk_mask=True, cond_mode="none",
                                         fold_idx=0, n_folds=5, use_mentor_split=False)
real_norm, gen_masked, gen_raw, real_suvr, gen_suvr, _ = \
    ev.load_cached_generations(test_ds, "none", a.generated_dir)
dk = ev._dk86_mask(test_ds); lab = ev._dk86_label_volume(test_ds)
R = np.stack([r*dk for r in real_suvr]); G = np.stack([g*dk for g in gen_suvr]); N = R.shape[0]; m = dk

def pr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x)&np.isfinite(y)
    x, y = x[ok], y[ok]
    return float(pearsonr(x, y)[0]) if x.size >= 2 and x.std() > 1e-8 and y.std() > 1e-8 else np.nan

pooled = pr(R[:, m].ravel(), G[:, m].ravel())
Rr = np.array([[R[i][lab==k].mean() for k in range(1,87)] for i in range(N)])
Gg = np.array([[G[i][lab==k].mean() for k in range(1,87)] for i in range(N)])
regional = float(np.nanmean([pr(Rr[:,k], Gg[:,k]) for k in range(86)]))
subject = pr(np.nanmean(Rr,1), np.nanmean(Gg,1))
metrics = dict(tag=a.tag, N=int(N), pooled_voxel_R=pooled,
               regional_R_86_mean=regional, subject_R=subject)
os.makedirs(a.out_dir, exist_ok=True)
json.dump(metrics, open(os.path.join(a.out_dir, f"mentor_metrics_{a.tag}.json"), "w"), indent=2)
print(f"\n=== {a.tag}: MENTOR METRICS (SUVR, DK86, self-consistent) ===")
print(f"  N={N}  PooledVoxelR={pooled:.3f}  RegionalR(86)={regional:.3f}  SubjectR={subject:.3f}")

# glass brains from the SAME masked-SUVR volumes -> guaranteed to match the numbers above
run_mode("cond_b_" + a.tag, "spatial", a.out_dir, use_dk_mask=True,
         data=([r*dk for r in real_suvr], [g*dk for g in gen_suvr], real_norm, gen_masked, test_ds),
         rank_by="pearson")
print(f"Glass brains -> {a.out_dir}/pearson/")
