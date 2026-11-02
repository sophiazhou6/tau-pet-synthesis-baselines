import os, sys, inspect, tempfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import nibabel as nib
import nilearn.plotting as nlplot

# cohort_v2 (597 subjects, 120 held-out test) — run with the GitHub taugennet as TAUGENNET_ROOT
# (its data/raw links to raw_v2) so the test set and its order are the evaluator's.
ROOT = os.environ.get("TAUGENNET_ROOT", "/scratch/network/sz3962/tau-pet-synthesis-baselines/taugennet")
DUR  = "/scratch/network/sz3962/tau-denseunet-baseline/results"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.config import VOL_SHAPE
from src.dataset_final import unnormalize, build_dataloaders
import glass_brain_final as GB
from glass_brain_final import _compute_affine

DPI = getattr(GB, "DPI", 150)
FG  = "#111111"                      # dark text on white
CMAP, VMIN = "hot_r", 0.0            # Raj-lab convention (Anil): hot_r from 0, one "Tau SUVR" bar

def render_glass_white(vol, affine, cmap, vmin, vmax, threshold=1e-3):
    """Same as glass_brain_final._render_glass but light-themed.
    black_bg=False makes nilearn draw the brain outline DARK, so it stays
    visible on white. Data encoding (cmap, vmin, vmax) is untouched."""
    nii = nib.Nifti1Image(vol.astype(np.float32), affine=affine)
    fig = plt.figure(figsize=(13, 3.75), facecolor="white")   # one row of Anil's 13×7.5 GT/Pred figure
    disp = nlplot.plot_glass_brain(nii, figure=fig, display_mode="lyrz",   # Anil's four views
                                   colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax, symmetric_cbar=False,
                                   black_bg=False, threshold=threshold, title=None)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f: tmp = f.name
    disp.savefig(tmp, dpi=DPI); disp.close(); plt.close(fig)
    img = plt.imread(tmp); os.unlink(tmp)
    return img

# 3 CN / 3 MCI / 3 AD test subjects whose Dense-U-Net (final model) per-subject r is closest to their
# group's median, among subjects with a measured plasma p-tau217 (so TauGenNet can show the same people):
# make_composite_figure.py --run denseunet_paper_refit --pick-cases
# Passed as GLASS_RIDS="rid,rid,..." so subject IDs stay out of the repository.
RIDS = [int(r) for r in os.environ["GLASS_RIDS"].split(",")]
OUT  = f"{DUR}/figures/baselines/cohort_v2/glass_9case_whitebg"; os.makedirs(OUT, exist_ok=True)
V = [
 ("Dense-U-Net", f"{DUR}/generated/baselines/cohort_v2/denseunet_paper_refit/single", f"{DUR}/records/baselines/cohort_v2/denseunet_paper_refit/single"),
 ("TauGenNet", f"{DUR}/generated/baselines/cohort_v2/taugennet_paper/single", f"{DUR}/records/baselines/cohort_v2/taugennet_paper/single"),
]

sig = inspect.signature(build_dataloaders)
kw  = {k: v for k, v in [("use_controls_322", True), ("use_dk_mask", True)] if k in sig.parameters}
_, _, test_ds, *_ = build_dataloaders("atrophy", **kw)
print("len(test_ds) =", len(test_ds))
affine = _compute_affine(test_ds, VOL_SHAPE)

base = pd.read_csv(f"{V[0][2]}/per_subject_wholebrain.csv")
bidx = {int(r.RID): int(r.subject_idx.split("_")[1]) for r in base.itertuples()}
grp  = {int(r.RID): r.group for r in base.itertuples()}

reals, norms = {}, {}
for rid in RIDS:
    i = bidx[rid]; assert f"RID_{rid}" in test_ds.pet_paths[i], (rid, test_ds.pet_paths[i])
    pet = np.asarray(test_ds[i][0]).squeeze()
    pmin, pmax = test_ds.get_pet_norms(i)
    reals[rid] = unnormalize(pet, pmin, pmax); norms[rid] = (pmin, pmax, pet)

def load_gen(gdir, i):
    """Generated volume i (normalized); predictions may be loose .npy or packed predictions.npz."""
    f = f"{gdir}/subject_{i:03d}.npy"
    if os.path.exists(f): return np.load(f).squeeze()
    if os.path.exists(f"{gdir}/predictions.npz"):
        with np.load(f"{gdir}/predictions.npz") as z:
            return z[f"subject_{i:03d}"].squeeze() if f"subject_{i:03d}" in z.files else None
    return None

gens = {}
for name, gdir, rdir in V:
    d = pd.read_csv(f"{rdir}/per_subject_wholebrain.csv")
    m = {int(r.RID): int(r.subject_idx.split("_")[1]) for r in d.itertuples()}
    gens[name] = {}
    for rid in RIDS:
        g = load_gen(gdir, m[rid]) if rid in m else None
        if g is None: gens[name][rid] = None; continue
        pmin, pmax, petn = norms[rid]
        g = unnormalize(g, pmin, pmax); g[petn <= 0] = 0
        gens[name][rid] = g

# Anil's range: 0 → highest in-brain SUVR (real and generated, all cases) rounded up to 0.5, ≥ 1.5
vox  = np.concatenate([v[norms[k][2] > 0] for k, v in reals.items()] +
                      [v[norms[k][2] > 0] for gd in gens.values() for k, v in gd.items() if v is not None])
VMAX = max(float(np.ceil(vox.max() * 2) / 2), 1.5)
print(f"shared colour range: SUVR {VMIN:g}–{VMAX:g}")

real_img = {rid: render_glass_white(reals[rid], affine, CMAP, VMIN, VMAX, 1e-3) for rid in RIDS}
print("real row rendered")

def sheet(rows, path, title):
    """3×3 grid of cases — one row per diagnosis, 3 subjects across; each cell stacks the
    `rows` renders (GT over Pred, as in Anil's per-subject glass figure)."""
    groups = list(dict.fromkeys(grp[rid] for rid in RIDS))
    cases = [[rid for rid in RIDS if grp[rid] == g] for g in groups]
    ncol, nst = max(map(len, cases)), len(rows)
    aspect = next(iter(real_img.values())).shape[0] / next(iter(real_img.values())).shape[1]
    w, gap = 5.2, 0.35                                # cell width (in), gap between diagnosis rows
    hts = ([w * aspect] * nst + [gap]) * len(groups)
    fig = plt.figure(figsize=(w * ncol + 1.4, sum(hts) + 0.5), dpi=200, facecolor="white")
    gs = fig.add_gridspec(len(hts), ncol, height_ratios=hts, wspace=.03, hspace=0,
                          left=.06, right=.92, top=1 - 0.5 / (sum(hts) + 0.5), bottom=0)
    for gi, (g, rids) in enumerate(zip(groups, cases)):
        for c, rid in enumerate(rids):
            for s, (lbl, imgs) in enumerate(rows):
                a = fig.add_subplot(gs[gi * (nst + 1) + s, c]); a.axis("off")
                if imgs.get(rid) is not None: a.imshow(imgs[rid])
                else: a.text(.5, .5, "n/a", ha="center", va="center", color="#999999", fontsize=8)
                if s == 0: a.set_title(f"RID {rid}", color=FG, fontsize=9, pad=2)
                if c == 0:
                    a.text(-.02, .5, lbl, transform=a.transAxes, va="center", ha="right",
                           color=FG, fontsize=9, fontweight="bold")
                    if s == 0:
                        a.text(-.16, 1 - nst / 2, g, transform=a.transAxes, rotation=90, va="center",
                               ha="center", color=FG, fontsize=12, fontweight="bold")
    fig.suptitle(title, color=FG, fontsize=13, fontweight="bold")
    cax = fig.add_axes([.935, .2, .01, .6])
    cb = ColorbarBase(cax, cmap=plt.get_cmap(CMAP), norm=Normalize(vmin=VMIN, vmax=VMAX), orientation="vertical")
    cb.set_label("Tau SUVR", color=FG, fontsize=11); cb.ax.tick_params(labelsize=9, colors=FG)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)

all_rows = [("GT", real_img)]
for name, gdir, rdir in V:
    imgs = {rid: None if g is None else render_glass_white(g, affine, CMAP, VMIN, VMAX, 1e-3)
            for rid, g in gens[name].items()}
    fn = name.lower().replace("-", "").replace(" ", "_")
    sheet([("GT", real_img), ("Pred", imgs)], f"{OUT}/glass2row_{fn}.png",
          f"Voxel Tau SUVR — GT vs Predicted — {name}, held-out test")
    all_rows.append((name, imgs))
    print(f"  wrote glass2row_{fn}.png  ({sum(v is not None for v in imgs.values())}/{len(RIDS)})")

if len(V) > 1:
    sheet(all_rows, f"{OUT}/glass_all_models.png", "Voxel Tau SUVR — GT vs every model, held-out test")
print("done ->", OUT)
