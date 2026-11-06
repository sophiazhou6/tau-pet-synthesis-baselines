#!/usr/bin/env python3
"""compare_glass.py -- side-by-side glass-brain montage across configs, shared SUVR scale.
Uses glass_brain_final's TAU_CMAP (blue->pink), BG, DPI so montages match the canonical
evaluate_final figures exactly. Does NOT reuse _load_test_data (legacy-split only).
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glass_brain_final import _compute_affine, _render_glass
try:
    from glass_brain_final import TAU_CMAP, BG, FG, DPI
except ImportError:
    TAU_CMAP = LinearSegmentedColormap.from_list("tau_pet",
        ["#08306b","#2171b5","#9ecae1","#fbb4b9","#f768a1","#ae017e"])
    BG, FG, DPI = "#0d0d0d", "#e8e8e8", 150
try:
    from glass_brain_final import unnormalize
except ImportError:
    def unnormalize(v, lo, hi): return v*(hi-lo)+lo

ap = argparse.ArgumentParser()
ap.add_argument("--panel", action="append", required=True, help="LABEL:DIR (repeatable)")
ap.add_argument("--subjects", type=int, nargs="+", default=[0,1,2])
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--n-folds", type=int, default=5)
ap.add_argument("--cond-mode", default="ptau217")
ap.add_argument("--cmap", default=None, help="override; default = canonical TAU_CMAP")
ap.add_argument("--out", required=True)
a = ap.parse_args()
cmap = TAU_CMAP if a.cmap is None else a.cmap

from src.dataset_spatial import build_dataloaders
out = build_dataloaders(cond_mode=a.cond_mode, use_dk_mask=True,
                        use_mentor_split=True, fold_idx=a.fold, n_folds=a.n_folds)
test_ds = out[2]
print("test subjects: %d" % len(test_ds))

panels = [s.split(":",1) for s in a.panel]
reals, norms = {}, {}
for i in a.subjects:
    real_np = test_ds[i][0].squeeze(0).numpy()
    pmin, pmax = test_ds.get_pet_norms(i)
    norms[i] = (pmin, pmax, real_np)
    r = unnormalize(real_np, pmin, pmax); r[real_np<=0]=0
    reals[i] = r
gens = {}
for label, d in panels:
    for i in a.subjects:
        p = os.path.join(d, "subject_%03d.npy" % i)
        if not os.path.exists(p):
            print("  MISSING %s" % p); continue
        pmin, pmax, real_np = norms[i]
        g = unnormalize(np.load(p), pmin, pmax); g[real_np<=0]=0
        gens[(label,i)] = g

affine = _compute_affine(test_ds, reals[a.subjects[0]].shape)
pool = np.concatenate([v[v>0] for v in list(reals.values())+list(gens.values())])
vmin, vmax = 0.0, float(np.percentile(pool, 99))
print("shared SUVR scale: vmin=%.2f vmax=%.2f  (~5-7 healthy; ~20 = normalization bug)" % (vmin,vmax))

cols = ["REAL"] + [l for l,_ in panels]
fig, axes = plt.subplots(len(a.subjects), len(cols),
                         figsize=(6.0*len(cols), 2.9*len(a.subjects)),
                         facecolor=BG, squeeze=False)
for r,i in enumerate(a.subjects):
    for c,label in enumerate(cols):
        ax = axes[r][c]
        vol = reals[i] if label=="REAL" else gens.get((label,i))
        if vol is None: ax.axis("off"); continue
        ax.imshow(_render_glass(vol, affine, cmap, vmin, vmax))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if r==0: ax.set_title(label, color=FG, fontsize=13)
        if c==0: ax.set_ylabel("subj %03d" % i, color=FG, fontsize=10)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=DPI, bbox_inches="tight", facecolor=BG)
print("wrote %s" % a.out)
