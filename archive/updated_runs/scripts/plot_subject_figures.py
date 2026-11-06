#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Per-subject ground-truth vs predicted tau figures for the held-out test set, in the Raj-lab style
(A. Kamat's plot_brain_slices_and_glass / plot_regional_brain_viz, Oct 2026), so every model's
figures can be compared side by side with the lab's own:

  voxel_tau_axial_slices_RID_<rid>.png               GT row over predicted row, axial z = -25…50 mm
  voxel_tau_glass_brain_RID_<rid>.png                GT over predicted, glass brain 'lyrz'
  regional_suvr_RID_<rid>_Slices_Comparison.png      the same for regional (DK86) mean SUVR
  regional_suvr_RID_<rid>_GlassBrain_Comparison.png

Ground truth is the subject's real tau PET clipped at 0, masked to the DK86 atlas brain and
resampled onto the atlas grid; the prediction is the exported SUVR NIfTI resampled onto the same
grid (NaN outside the atlas). Regional values are the evaluator's own regional means
(regional_means_{real,gen}_atrophy.csv), painted back onto the atlas labels they came from.
Colour range: 0 → max(GT, prediction) rounded up to 0.5, at least 1.5, as in the lab code.

    python scripts/plot_subject_figures.py --runs denseunet_paper --limit 3            # a quick look
    python scripts/plot_subject_figures.py --runs denseunet_paper taugennet_paper      # shared colour range
"""
import argparse
import inspect
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colorbar import ColorbarBase  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from nibabel.processing import resample_from_to  # noqa: E402
from nilearn import datasets, plotting  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.config import FIGURES_ROOT, GEN_ROOT, RECORDS_ROOT, RESULTS_DIR  # noqa: E402

ATLAS = os.path.join(os.environ.get("TAUGENNET_ROOT", "/scratch/network/sz3962/taugennet"),
                     "data", "raw_v2", "mni_dk_atlas.nii.gz")

# The lab snippet references these without defining them; values follow its regional maps
# ('hot_r', vmin 0) or are publication defaults. Change them here to match the lab exactly.
BRAIN_CMAP = "hot_r"
BRAIN_VMIN = 0.0
BRAIN_VMAX = 3.5          # only used when a subject has no finite voxels
CUT_Z = [-25, -10, 5, 20, 35, 50]
PRETTY = {"denseunet": "Dense-U-Net", "denseunet_paper": "Dense-U-Net", "denseunet_paper_refit": "Dense-U-Net", "taugennet_paper": "TauGenNet", "swinunetr": "Swin UNETR", "eagan": "gEa-GAN", "deagan": "dEa-GAN",
          "ct2pet": "Synthetic-PET cGAN", "jointsyn": "Joint synthesis+diagnosis",
          "jointsyn_paired": "Joint synthesis+diagnosis (paired)"}
# nilearn ≥ 0.12 ignores add_overlay(alpha=…) in favour of transparency=…, and plot_anat draws a
# colorbar unless told not to; the lab code predates both, so pass whichever this version expects.
_OVERLAY_ALPHA = ({"transparency": 0.7} if "transparency" in inspect.signature(
    plotting.displays._slicers.BaseSlicer.add_overlay).parameters else {"alpha": 0.7})
_ANAT_KW = {"colorbar": False} if "colorbar" in inspect.signature(plotting.plot_anat).parameters else {}
PUB_FONT, PUB_TICK, PUB_TITLE = 18, 14, 20
DPI = 200


def dyn_vmax(*arrays, fallback=BRAIN_VMAX):
    v = np.concatenate([a[np.isfinite(a)] for a in arrays])
    vmax = float(np.ceil(float(v.max()) * 2) / 2) if v.size else fallback
    return max(vmax, 1.5)


def glass_kw(vmax):
    return {"vmax": vmax, "vmin": BRAIN_VMIN, "symmetric_cbar": False, "black_bg": False}


def slices_figure(gt_img, pred_img, bg, vmax, figsize, hspace, ylabel_size, cbar_rect, cbar_label,
                  cbar_size, title, title_size, path):
    n = len(CUT_Z)
    fig, axes = plt.subplots(2, n, figsize=figsize)
    fig.subplots_adjust(hspace=hspace, wspace=0.04)
    for col, z in enumerate(CUT_Z):
        for row, (img, lbl) in enumerate([(gt_img, "GT"), (pred_img, "Pred")]):
            display = plotting.plot_anat(bg, display_mode="z", cut_coords=[z], axes=axes[row, col],
                                         draw_cross=False, annotate=False, black_bg=False, dim=-0.5,
                                         threshold=1e-3, **_ANAT_KW)
            display.add_overlay(img, cmap=BRAIN_CMAP, threshold=1e-3, colorbar=False,
                                vmin=BRAIN_VMIN, vmax=vmax, **_OVERLAY_ALPHA)
            if col == 0:                 # nilearn hides the host axes; re-show it for the row label
                ax = axes[row, col]
                ax.set_axis_on(); ax.set_xticks([]); ax.set_yticks([])
                ax.patch.set_visible(False)  # its opaque background would otherwise cover the slice
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_ylabel(lbl, fontsize=ylabel_size, fontweight="bold")
    for col, z in enumerate(CUT_Z):
        axes[1, col].text(0.5, -0.06, f"z = {z} mm", transform=axes[1, col].transAxes,
                          ha="center", va="top", fontsize=PUB_TICK, color="black")
    cbar_ax = fig.add_axes(cbar_rect)
    ColorbarBase(cbar_ax, cmap=plt.get_cmap(BRAIN_CMAP), norm=Normalize(vmin=BRAIN_VMIN, vmax=vmax),
                 orientation="vertical").set_label(cbar_label, fontsize=cbar_size)
    fig.suptitle(title, fontsize=title_size, fontweight="bold")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def glass_figure(gt_img, pred_img, vmax, figsize, cbar_label, cbar_size, title, title_size, path):
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.subplots_adjust(hspace=0.05, top=0.91, right=0.88)
    for ax, img, lbl in zip(axes, [gt_img, pred_img], ["GT", "Pred"]):
        plotting.plot_glass_brain(img, display_mode="lyrz", colorbar=False, title=None, cmap=BRAIN_CMAP,
                                  axes=ax, figure=fig, threshold=1e-3, **glass_kw(vmax))
        box = ax.get_position()
        fig.text(box.x0 - 0.005, (box.y0 + box.y1) / 2, lbl, rotation=90, ha="right", va="center",
                 fontsize=PUB_FONT - 4, fontweight="bold")
    cbar_ax = fig.add_axes([0.90, 0.15, 0.013, 0.70])
    ColorbarBase(cbar_ax, cmap=plt.get_cmap(BRAIN_CMAP), norm=Normalize(vmin=BRAIN_VMIN, vmax=vmax),
                 orientation="vertical").set_label(cbar_label, fontsize=cbar_size)
    fig.suptitle(title, fontsize=title_size, fontweight="bold")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def load_run(cohort, run):
    """run = "<tag>" or "<tag>:<fold dir>" (default fold dir "single", i.e. a non-CV run)."""
    tag, fn = run.split(":", 1) if ":" in run else (run, "single")
    rec = os.path.join(RECORDS_ROOT, cohort, tag, fn)
    order = pd.read_csv(os.path.join(GEN_ROOT, cohort, tag, fn, "index.csv"))
    return dict(tag=tag, fn=fn, pred_dir=os.path.join(RESULTS_DIR, "export_nifti", cohort, tag, fn, "test"),
                row=dict(zip(order.RID.astype(str), order["index"].astype(int))), pet=dict(zip(order.RID.astype(str), order.pet)),
                reg_real=pd.read_csv(os.path.join(rec, "regional_means_real_atrophy.csv"), index_col=0),
                reg_gen=pd.read_csv(os.path.join(rec, "regional_means_gen_atrophy.csv"), index_col=0))


def run_subjects(cohort, runs, kinds, rids, limit, out_root, atlas_img, atlas, bg):
    """Every model's figures for one subject share the colour range: max over GT and all predictions."""
    grid = (atlas.shape, atlas_img.affine)
    inside = atlas > 0
    brain = nib.Nifti1Image(inside.astype(np.float32), atlas_img.affine)

    def paint(values):
        vol = np.zeros(atlas.shape, dtype=np.float32)
        for k, val in enumerate(values, start=1):           # column k-1 was measured on label k
            if np.isfinite(val):
                vol[atlas == k] = val
        return nib.Nifti1Image(vol, atlas_img.affine)

    common = set.intersection(*(set(r["row"]) for r in runs))
    todo = [x for x in runs[0]["row"] if x in common and (not rids or x in rids)]
    todo = todo[:limit] if limit else todo
    for r in runs:
        os.makedirs(os.path.join(out_root, r["tag"], r["fn"]), exist_ok=True)
    for rid in todo:
        if "voxel" in kinds:
            tau = nib.load(runs[0]["pet"][rid])
            gt = np.clip(np.asarray(tau.dataobj, dtype=np.float32), 0, None)
            gt[np.asarray(resample_from_to(brain, tau, order=0).dataobj) < 0.5] = 0.0
            ga = np.asarray(resample_from_to(nib.Nifti1Image(gt, tau.affine), grid, order=1).dataobj, dtype=np.float32).copy()
            ga[~inside] = 0.0                                 # same support as the prediction panels
            gt_img = nib.Nifti1Image(ga, atlas_img.affine)
            preds = {}
            for r in runs:
                pa = np.asarray(resample_from_to(nib.load(os.path.join(r["pred_dir"], f"RID_{rid}_pred_tauPET_SUVR.nii.gz")),
                                                 grid, order=1).dataobj, dtype=np.float32).copy()
                pa[~inside] = np.nan
                preds[r["tag"]] = pa
            vmax = dyn_vmax(ga[inside], *[p[inside] for p in preds.values()])
            for r in runs:
                out_dir = os.path.join(out_root, r["tag"], r["fn"])
                pred_img = nib.Nifti1Image(preds[r["tag"]], atlas_img.affine)
                title = f"Voxel Tau SUVR — GT vs Predicted ({PRETTY.get(r['tag'], r['tag'])}) — RID {rid}"
                try:
                    slices_figure(gt_img, pred_img, bg, vmax, (3.0 * len(CUT_Z), 6.5), 0.12, PUB_FONT - 4,
                                  [0.92, 0.15, 0.012, 0.70], "Tau SUVR", PUB_FONT - 2, title, PUB_TITLE,
                                  os.path.join(out_dir, f"voxel_tau_axial_slices_RID_{rid}.png"))
                except Exception as e:                      # noqa: BLE001
                    plt.close("all"); print(f"  [WARN] RID {rid} {r['tag']} - axial slices failed: {e}")
                try:
                    glass_figure(gt_img, pred_img, vmax, (13, 7.5), "Tau SUVR", PUB_FONT - 2, title, PUB_TITLE,
                                 os.path.join(out_dir, f"voxel_tau_glass_brain_RID_{rid}.png"))
                except Exception as e:                      # noqa: BLE001
                    plt.close("all"); print(f"  [WARN] RID {rid} {r['tag']} - glass brain failed: {e}")
        if "regional" in kinds:
            g = runs[0]["reg_real"].loc[f"subject_{runs[0]['row'][rid]:03d}"].to_numpy(float)
            ps = {r["tag"]: r["reg_gen"].loc[f"subject_{r['row'][rid]:03d}"].to_numpy(float) for r in runs}
            vmax = dyn_vmax(g, *ps.values())
            for r in runs:
                out_dir = os.path.join(out_root, r["tag"], r["fn"])
                title = f"Regional SUVR — GT vs Predicted ({PRETTY.get(r['tag'], r['tag'])}) — RID {rid}"
                try:
                    slices_figure(paint(g), paint(ps[r["tag"]]), bg, vmax, (3.5 * len(CUT_Z), 7.5), 0.06, PUB_FONT - 2,
                                  [0.92, 0.15, 0.013, 0.70], "Regional SUVR", PUB_FONT, title, PUB_TITLE + 2,
                                  os.path.join(out_dir, f"regional_suvr_RID_{rid}_Slices_Comparison.png"))
                except Exception as e:                      # noqa: BLE001
                    plt.close("all"); print(f"  [WARN] RID {rid} {r['tag']} - regional slices failed: {e}")
                try:
                    glass_figure(paint(g), paint(ps[r["tag"]]), vmax, (14, 9), "Regional SUVR", PUB_FONT, title,
                                 PUB_TITLE + 2, os.path.join(out_dir, f"regional_suvr_RID_{rid}_GlassBrain_Comparison.png"))
                except Exception as e:                      # noqa: BLE001
                    plt.close("all"); print(f"  [WARN] RID {rid} {r['tag']} - regional glass-brain failed: {e}")
    print(f"[figures] {len(todo)} subjects × {len(runs)} model(s) → {out_root}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="cohort_v2")
    ap.add_argument("--runs", nargs="+", required=True,
                    help='"<tag>" (non-CV run) or "<tag>:fold_<k>"; all runs of one call share each subject\'s colour range')
    ap.add_argument("--kinds", nargs="+", default=["voxel", "regional"], choices=["voxel", "regional"])
    ap.add_argument("--rids", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=None, help="default: results/figures/baselines/<cohort>/per_subject_rajlab")
    a = ap.parse_args()
    out_root = a.out_root or os.path.join(FIGURES_ROOT, a.cohort, "per_subject_rajlab")
    atlas_img = nib.load(ATLAS)
    atlas = np.rint(np.asarray(atlas_img.dataobj)).astype(np.int32)
    bg = datasets.load_mni152_template(resolution=1)
    runs = [load_run(a.cohort, r) for r in a.runs]
    run_subjects(a.cohort, runs, a.kinds, set(a.rids) if a.rids else None, a.limit, out_root, atlas_img, atlas, bg)


if __name__ == "__main__":
    main()
