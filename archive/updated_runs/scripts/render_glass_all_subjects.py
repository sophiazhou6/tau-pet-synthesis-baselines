#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Glass-brain figures for every held-out test subject, so a subject can be picked for the paper.

Same rendering as panel A of the composite figures (analysis/render_glass_9case_white.py): nilearn glass
brain 'lyrz', hot_r from 0, one "Tau SUVR" colour bar, and the colour range rule 0 → highest in-brain SUVR
rounded up to 0.5 (at least 1.5), here per subject and shared by GT and both models. Predictions are
converted to SUVR the way the evaluator does (each subject's real PET min/max) and masked to the DK86 atlas.

Writes to results/figures/baselines/cohort_v2/glass_per_subject/:
  DenseUNet/RID_<rid>_<group>_glass_GT_vs_DenseUNet.png     GT over prediction, the lab's 13 x 7.5 layout
  TauGenNet/RID_<rid>_<group>_glass_GT_vs_TauGenNet.png
  all_test_subjects_GT_DenseUNet_TauGenNet.pdf              one page per subject: GT, Dense-U-Net, TauGenNet
  index.csv                                                 page, RID, group, each model's subject-level r
"""
import os
import re
import sys
import tempfile
from multiprocessing import Pool

GH = "/scratch/network/sz3962/tau-pet-synthesis-baselines/taugennet"
os.environ["TAUGENNET_ROOT"] = GH
sys.path[:0] = [GH, os.path.join(GH, "scripts")]

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import nilearn.plotting as nlplot  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colorbar import ColorbarBase  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from PIL import Image  # noqa: E402

from glass_brain_final import _compute_affine  # noqa: E402
from src.config import VOL_SHAPE  # noqa: E402
from src.dataset_final import build_dataloaders, unnormalize  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT = os.path.join(RESULTS, "figures", "baselines", "cohort_v2", "glass_per_subject")
MODELS = [("Dense-U-Net", "DenseUNet", "denseunet_paper_refit"), ("TauGenNet", "TauGenNet", "taugennet_paper")]
CMAP, VMIN, DPI, FG = "hot_r", 0.0, 150, "#111111"
PLASMA = os.path.join(GH, "data", "raw", "plasma_imputed", "refit.csv")
S = {}                                          # filled before the workers fork


def render_row(vol, vmax):
    """One row of the lab's GT/Pred glass figure (13 x 3.75 in), as an image array."""
    nii = nib.Nifti1Image(vol.astype(np.float32), affine=S["affine"])
    fig = plt.figure(figsize=(13, 3.75), facecolor="white")
    disp = nlplot.plot_glass_brain(nii, figure=fig, display_mode="lyrz", colorbar=False, cmap=CMAP, vmin=VMIN,
                                   vmax=vmax, symmetric_cbar=False, black_bg=False, threshold=1e-3, title=None)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    disp.savefig(tmp, dpi=DPI)
    disp.close()
    plt.close(fig)
    img = plt.imread(tmp)
    os.unlink(tmp)
    return img


def sheet(rows, vmax, title, path, dpi=DPI, left=0.07):
    """Rendered rows stacked with their labels on the left and one colour bar on the right."""
    fig = plt.figure(figsize=(13, 3.75 * len(rows) + 0.6), facecolor="white")
    h = 3.75 / (3.75 * len(rows) + 0.6)
    for k, (label, img) in enumerate(rows):
        ax = fig.add_axes([left, 1 - 0.6 / (3.75 * len(rows) + 0.6) - (k + 1) * h, 0.91 - left, h])
        ax.imshow(img)
        ax.axis("off")
        ax.text(-0.01, 0.5, label, transform=ax.transAxes, ha="right", va="center", color=FG, fontsize=13,
                fontweight="bold")
    cax = fig.add_axes([0.93, 0.15, 0.012, 0.65])
    cb = ColorbarBase(cax, cmap=plt.get_cmap(CMAP), norm=Normalize(vmin=VMIN, vmax=vmax), orientation="vertical")
    cb.set_label("Tau SUVR", color=FG, fontsize=12)
    cb.ax.tick_params(labelsize=10, colors=FG)
    fig.suptitle(title, color=FG, fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def one_subject(i):
    rid, grp, real = S["rids"][i], S["groups"][i], S["real"][i]
    preds = {name: S["pred"][name][i] for name, _, _ in MODELS}
    inside = S["mask"]
    vmax = max(float(np.ceil(max(real[inside].max(), *[p[inside].max() for p in preds.values()]) * 2) / 2), 1.5)
    gt_img = render_row(real, vmax)
    pred_img = {name: render_row(p, vmax) for name, p in preds.items()}
    for name, short, _ in MODELS:
        sheet([("GT", gt_img), ("Pred", pred_img[name])], vmax,
              f"Voxel Tau SUVR — GT vs Predicted ({name}) — RID {rid}",
              os.path.join(OUT, short, f"RID_{rid}_{grp}_glass_GT_vs_{short}.png"))
    r = {name: S["r"][name][rid] for name, _, _ in MODELS}
    page = os.path.join(S["tmp"], f"page_{i:03d}.png")
    sheet([("GT", gt_img)] + [(f"{name}\nr = {r[name]:.2f}", pred_img[name]) for name, _, _ in MODELS], vmax,
          f"RID {rid} · {grp} · plasma {'imputed' if S['imputed'][rid] else 'measured'}", page, dpi=110, left=0.15)
    return i, page


def main():
    for _, short, _ in MODELS:
        os.makedirs(os.path.join(OUT, short), exist_ok=True)
    _, _, te, *_ = build_dataloaders("atrophy", use_controls_322=True, use_dk_mask=True)
    mask_t = te._dk_mask
    mask = mask_t.squeeze().numpy() > 0
    rids = [int(re.search(r"RID[_-](\d+)", p).group(1)) for p in te.pet_paths]
    real, norms = [], []
    for p in te.pet_paths:
        v, lo, hi = te._load(p, mask_t)
        real.append(unnormalize(v.squeeze().numpy(), lo, hi) * mask)
        norms.append((lo, hi))
    pred, rr, groups = {}, {}, None
    for name, _, tag in MODELS:
        gen = os.path.join(RESULTS, "generated", "baselines", "cohort_v2", tag, "single")
        if [int(x) for x in pd.read_csv(os.path.join(gen, "index.csv")).RID] != rids:
            sys.exit(f"{tag}: generated order differs from the test set")
        with np.load(os.path.join(gen, "predictions.npz")) as z:
            pred[name] = [unnormalize(z[f"subject_{i:03d}"].squeeze(), lo, hi) * mask for i, (lo, hi) in enumerate(norms)]
        ps = pd.read_csv(os.path.join(RESULTS, "records", "baselines", "cohort_v2", tag, "single", "per_subject_wholebrain.csv"))
        rr[name] = dict(zip(ps.RID.astype(int), ps.pearson_wholebrain))
        groups = dict(zip(ps.RID.astype(int), ps.group))
    imputed = {int(r.RID): int(r.imputed) for r in pd.read_csv(PLASMA).itertuples()}
    order = sorted(range(len(rids)), key=lambda i: (["CN", "MCI", "AD"].index(groups[rids[i]]), rids[i]))
    S.update(affine=_compute_affine(te, VOL_SHAPE), mask=mask, rids=rids, groups=[groups[r] for r in rids],
             real=real, pred=pred, r=rr, imputed=imputed, tmp=tempfile.mkdtemp(dir=OUT))
    with Pool(int(os.environ.get("N_WORKERS", 6))) as pool:
        pages = dict(pool.imap_unordered(one_subject, order))
        print(f"rendered {len(pages)} subjects", flush=True)
    imgs = [Image.open(pages[i]).convert("RGB") for i in order]
    pdf = os.path.join(OUT, "all_test_subjects_GT_DenseUNet_TauGenNet.pdf")
    imgs[0].save(pdf, save_all=True, append_images=imgs[1:], resolution=110, quality=85)
    for p in pages.values():
        os.unlink(p)
    os.rmdir(S["tmp"])
    pd.DataFrame([dict(page=k + 1, RID=rids[i], group=groups[rids[i]],
                       plasma="imputed" if imputed[rids[i]] else "measured",
                       **{f"r_{short}": round(rr[name][rids[i]], 3) for name, short, _ in MODELS})
                  for k, i in enumerate(order)]).to_csv(os.path.join(OUT, "index.csv"), index=False)
    print(f"→ {OUT}  ({os.path.getsize(pdf) / 1e6:.0f} MB PDF)")


if __name__ == "__main__":
    main()
