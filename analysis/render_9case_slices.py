import os, sys, inspect
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.environ.get("TAUGENNET_ROOT", os.path.expanduser("~/taugennet"))
DUR  = os.path.expanduser("~/tau-denseunet-baseline/results")
sys.path.insert(0, ROOT)
from src.dataset_final import unnormalize
from src.dataset_spatial import build_dataloaders

TAU_CMAP = LinearSegmentedColormap.from_list(
    "tau_pet", ["#08306b","#2171b5","#9ecae1","#fbb4b9","#f768a1","#ae017e"])
RIDS = [6516, 6449, 6374, 6911, 6880, 6702, 6545, 6840, 6976]
OUT  = os.path.expanduser("~/fig_9case_slices"); os.makedirs(OUT, exist_ok=True)

VARIANTS = [
 ("DU_plain",     f"{DUR}/generated/controls322_plain_v2",
                  f"{DUR}/records/denseunet_baseline/controls322_plain_v2"),
 ("DU_atrophy",   f"{DUR}/generated/controls322_paint_atrophy",
                  f"{DUR}/records/denseunet_baseline/controls322_paint_atrophy"),
 ("DU_suvr",      f"{DUR}/generated/controls322_suvr_lr1e-4_wd0",
                  f"{DUR}/records/denseunet_baseline/controls322_suvr_lr1e-4_wd0"),
 ("DU_both",      f"{DUR}/generated/controls322_paint_both",
                  f"{DUR}/records/denseunet_baseline/controls322_paint_both"),
 ("TG_atrophy",   f"{ROOT}/results/generated/controls_322_TEST/final_t1_atrophy_enc",
                  f"{ROOT}/results/records/controls_322_TEST/final_t1_atrophy_enc"),
 ("TG_suvr",      f"{ROOT}/results/generated/controls_322_TEST/final_t2_suvr_enc",
                  f"{ROOT}/results/records/controls_322_TEST/final_t2_suvr_enc"),
 ("TG_both",      f"{ROOT}/results/generated/controls_322_TEST/final_t2b_both_enc",
                  f"{ROOT}/results/records/controls_322_TEST/final_t2b_both_enc"),
 ("TG_plasma",    f"{ROOT}/results/generated/controls_322_TEST/final_t3_suvr_enc_plasma",
                  f"{ROOT}/results/records/controls_322_TEST/final_t3_suvr_enc_plasma"),
]

sig = inspect.signature(build_dataloaders)
kw  = {k: v for k, v in [("use_controls_322", True), ("use_dk_mask", True),
                         ("cond_mode", "none")] if k in sig.parameters}
_, _, test_ds, *_ = build_dataloaders(**kw)
print("len(test_ds) =", len(test_ds), "(expect 64)")

base = pd.read_csv(f"{ROOT}/results/records/controls_322_TEST/final_t1_atrophy_enc/per_subject_wholebrain.csv")
bidx = {int(r.RID): int(r.subject_idx.split("_")[1]) for r in base.itertuples()}
miss = [r for r in RIDS if r not in bidx]
if miss: sys.exit(f"RIDs not in the 64-subject TEST set: {miss}")

reals, mris, norms = {}, {}, {}
for rid in RIDS:
    i = bidx[rid]
    it = test_ds[i]
    pet, mri = np.asarray(it[0]).squeeze(), np.asarray(it[1]).squeeze()
    pmin, pmax = test_ds.get_pet_norms(i)
    reals[rid], mris[rid], norms[rid] = unnormalize(pet, pmin, pmax), mri, (pmin, pmax, pet)

gens = {}
for name, gdir, rdir in VARIANTS:
    csvp = f"{rdir}/per_subject_wholebrain.csv"
    if not os.path.exists(csvp):
        print(f"  !! no per-subject CSV for {name}, skipping"); continue
    d = pd.read_csv(csvp)
    m = {int(r.RID): int(r.subject_idx.split("_")[1]) for r in d.itertuples()}
    n_ok = 0
    for rid in RIDS:
        if rid not in m: continue
        f = f"{gdir}/subject_{m[rid]:03d}.npy"
        if not os.path.exists(f): continue
        pmin, pmax, petn = norms[rid]
        g = unnormalize(np.load(f).squeeze(), pmin, pmax); g[petn <= 0] = 0
        gens[(name, rid)] = g; n_ok += 1
    print(f"  {name:12s} n_subjects={len(d):3d}  rendered {n_ok}/8")

pet_vmax = float(np.percentile(np.concatenate([v[v>0].ravel() for v in reals.values()]), 99))
ma = np.concatenate([v[v>0].ravel() for v in mris.values()])
mvmin, mvmax = float(np.percentile(ma,1)), float(np.percentile(ma,99))
Z = reals[RIDS[0]].shape[2] // 2
print(f"shared pet_vmax={pet_vmax:.3f}  z={Z}")

def save(a, path, cmap, vmin, vmax):
    fig, ax = plt.subplots(figsize=(3,3), dpi=200)
    ax.imshow(a[:,:,Z].T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.axis("off"); fig.subplots_adjust(0,0,1,1)
    fig.savefig(path, bbox_inches="tight", pad_inches=0, facecolor="#0d0d0d"); plt.close(fig)

for rid in RIDS:
    save(mris[rid],  f"{OUT}/RID{rid}_mri.png",  "gray", mvmin, mvmax)
    save(reals[rid], f"{OUT}/RID{rid}_real.png", TAU_CMAP, 0.0, pet_vmax)
for (name, rid), g in gens.items():
    save(g, f"{OUT}/RID{rid}_{name}.png", TAU_CMAP, 0.0, pet_vmax)

rows = ["MRI","Real"] + [v[0] for v in VARIANTS]
fig, axes = plt.subplots(len(rows), len(RIDS),
                         figsize=(1.6*len(RIDS), 1.6*len(rows)), dpi=200,
                         facecolor="#0d0d0d")
for r, rn in enumerate(rows):
    for c, rid in enumerate(RIDS):
        ax = axes[r, c]; ax.axis("off")
        if rn == "MRI":    ax.imshow(mris[rid][:,:,Z].T, cmap="gray", vmin=mvmin, vmax=mvmax, origin="lower")
        elif rn == "Real": ax.imshow(reals[rid][:,:,Z].T, cmap=TAU_CMAP, vmin=0, vmax=pet_vmax, origin="lower")
        elif (rn, rid) in gens:
            ax.imshow(gens[(rn,rid)][:,:,Z].T, cmap=TAU_CMAP, vmin=0, vmax=pet_vmax, origin="lower")
        else:
            ax.text(.5,.5,"n/a",ha="center",va="center",color="#777",fontsize=7)
        if r == 0: ax.set_title(f"RID {rid}", color="#e8e8e8", fontsize=7)
        if c == 0: ax.text(-0.12,.5,rn,transform=ax.transAxes,rotation=90,
                           va="center",ha="center",color="#e8e8e8",fontsize=7)
fig.subplots_adjust(wspace=.02, hspace=.02)
fig.savefig(f"{OUT}/grid_all_variants.png", bbox_inches="tight", facecolor="#0d0d0d")
print(f"wrote {len(gens)+16} PNGs + grid_all_variants.png to {OUT}")
