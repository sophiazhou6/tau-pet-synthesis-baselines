import os, sys, inspect, tempfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.plotting as nlplot

ROOT = os.environ.get("TAUGENNET_ROOT", os.path.expanduser("~/taugennet"))
DUR  = os.path.expanduser("~/tau-denseunet-baseline/results")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.config import VOL_SHAPE
from src.dataset_final import unnormalize
from src.dataset_spatial import build_dataloaders
import glass_brain_final as GB
from glass_brain_final import TAU_CMAP, _compute_affine

DPI = getattr(GB, "DPI", 150)
FG  = "#111111"                      # dark text on white

def render_glass_white(vol, affine, cmap, vmin, vmax, threshold=1e-3):
    """Same as glass_brain_final._render_glass but light-themed.
    black_bg=False makes nilearn draw the brain outline DARK, so it stays
    visible on white. Data encoding (cmap, vmin, vmax) is untouched."""
    nii = nib.Nifti1Image(vol.astype(np.float32), affine=affine)
    fig = plt.figure(figsize=(8, 3.5), facecolor="white")
    disp = nlplot.plot_glass_brain(nii, figure=fig, display_mode="ortho",
                                   colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
                                   black_bg=False, threshold=threshold, annotate=False)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f: tmp = f.name
    disp.savefig(tmp, dpi=DPI); disp.close(); plt.close(fig)
    img = plt.imread(tmp); os.unlink(tmp)
    return img

RIDS = [6516, 6449, 6374, 6911, 6880, 6702, 6545, 6840, 6976]
OUT  = os.path.expanduser("~/fig_glass_9case_whitebg"); os.makedirs(OUT, exist_ok=True)
V = [
 ("DU_plain",   f"{DUR}/generated/controls322_plain_v2",        f"{DUR}/records/denseunet_baseline/controls322_plain_v2"),
 ("DU_atrophy", f"{DUR}/generated/controls322_paint_atrophy",   f"{DUR}/records/denseunet_baseline/controls322_paint_atrophy"),
 ("DU_suvr",    f"{DUR}/generated/controls322_suvr_lr1e-4_wd0", f"{DUR}/records/denseunet_baseline/controls322_suvr_lr1e-4_wd0"),
 ("DU_both",    f"{DUR}/generated/controls322_paint_both",      f"{DUR}/records/denseunet_baseline/controls322_paint_both"),
 ("TG_atrophy", f"{ROOT}/results/generated/controls_322_TEST/final_t1_atrophy_enc",     f"{ROOT}/results/records/controls_322_TEST/final_t1_atrophy_enc"),
 ("TG_suvr",    f"{ROOT}/results/generated/controls_322_TEST/final_t2_suvr_enc",        f"{ROOT}/results/records/controls_322_TEST/final_t2_suvr_enc"),
 ("TG_both",    f"{ROOT}/results/generated/controls_322_TEST/final_t2b_both_enc",       f"{ROOT}/results/records/controls_322_TEST/final_t2b_both_enc"),
 ("TG_plasma",  f"{ROOT}/results/generated/controls_322_TEST/final_t3_suvr_enc_plasma", f"{ROOT}/results/records/controls_322_TEST/final_t3_suvr_enc_plasma"),
]

sig = inspect.signature(build_dataloaders)
kw  = {k: v for k, v in [("use_controls_322", True), ("use_dk_mask", True),
                         ("cond_mode", "none")] if k in sig.parameters}
_, _, test_ds, *_ = build_dataloaders(**kw)
print("len(test_ds) =", len(test_ds))
affine = _compute_affine(test_ds, VOL_SHAPE)

base = pd.read_csv(f"{ROOT}/results/records/controls_322_TEST/final_t1_atrophy_enc/per_subject_wholebrain.csv")
bidx = {int(r.RID): int(r.subject_idx.split("_")[1]) for r in base.itertuples()}
grp  = {int(r.RID): r.group for r in base.itertuples()}

reals, norms = {}, {}
for rid in RIDS:
    i = bidx[rid]; pet = np.asarray(test_ds[i][0]).squeeze()
    pmin, pmax = test_ds.get_pet_norms(i)
    reals[rid] = unnormalize(pet, pmin, pmax); norms[rid] = (pmin, pmax, pet)
VMAX = float(np.percentile(np.concatenate([v[v > 0].ravel() for v in reals.values()]), 99))
print(f"shared pet_vmax = {VMAX:.3f}   (unchanged from the black-bg version)")

real_img = {rid: render_glass_white(reals[rid], affine, TAU_CMAP, 0.0, VMAX, 1e-3) for rid in RIDS}
print("real row rendered")

def sheet(rows, path, title):
    fig, ax = plt.subplots(len(rows), len(RIDS),
                           figsize=(2.2*len(RIDS), 1.9*len(rows)), dpi=200, facecolor="white")
    ax = np.atleast_2d(ax)
    for r, (lbl, imgs) in enumerate(rows):
        for c, rid in enumerate(RIDS):
            a = ax[r, c]; a.axis("off"); a.set_facecolor("white")
            if imgs.get(rid) is not None: a.imshow(imgs[rid])
            else: a.text(.5, .5, "n/a", ha="center", va="center", color="#999999", fontsize=8)
            if r == 0: a.set_title(f"RID {rid}\n{grp.get(rid,'')}", color=FG, fontsize=8)
            if c == 0: a.text(-.06, .5, lbl, transform=a.transAxes, rotation=90,
                              va="center", ha="center", color=FG, fontsize=8)
    fig.suptitle(title, color=FG, fontsize=10)
    fig.subplots_adjust(wspace=.02, hspace=.02, top=.90)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)

all_rows = [("Real", real_img)]
for name, gdir, rdir in V:
    d = pd.read_csv(f"{rdir}/per_subject_wholebrain.csv")
    m = {int(r.RID): int(r.subject_idx.split("_")[1]) for r in d.itertuples()}
    imgs = {}
    for rid in RIDS:
        f = f"{gdir}/subject_{m[rid]:03d}.npy" if rid in m else None
        if not f or not os.path.exists(f): imgs[rid] = None; continue
        pmin, pmax, petn = norms[rid]
        g = unnormalize(np.load(f).squeeze(), pmin, pmax); g[petn <= 0] = 0
        imgs[rid] = render_glass_white(g, affine, TAU_CMAP, 0.0, VMAX, 1e-3)
    sheet([("Real", real_img), ("Generated", imgs)], f"{OUT}/glass2row_{name}.png",
          f"{name}  —  real (top) vs generated (bottom),  SUVR 0–{VMAX:.2f}")
    all_rows.append((name, imgs))
    print(f"  wrote glass2row_{name}.png  ({sum(v is not None for v in imgs.values())}/{len(RIDS)})")

sheet(all_rows, f"{OUT}/glass_all_variants.png", f"Real + all 8 variants,  SUVR 0–{VMAX:.2f}")
print("done ->", OUT)
