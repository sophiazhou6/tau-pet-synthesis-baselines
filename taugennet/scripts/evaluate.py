import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

from src.config import DEVICE, VOL_SHAPE, FIGURES_DIR
from src import dataset_final as _dataset_v2
from src import dataset_combined as _dataset_combined
from src.diffusion import DiffusionSchedule
from src.inference import load_models, synthesize_tau_pet, synthesize_no_mri


# ── ROI definitions (approximate bounding boxes as fractions of VOL_SHAPE) ───
ROI_DEFS = {
    "Parahippocampal":   (0.40, 0.55, 0.35, 0.65, 0.30, 0.55),
    "Fusiform":          (0.45, 0.60, 0.30, 0.70, 0.25, 0.50),
    "Inferior Temporal": (0.30, 0.55, 0.25, 0.75, 0.20, 0.55),
    "Hippocampus":       (0.42, 0.52, 0.40, 0.60, 0.35, 0.50),
    "Post. Cingulate":   (0.40, 0.55, 0.38, 0.62, 0.50, 0.70),
    "Entorhinal":        (0.43, 0.55, 0.38, 0.62, 0.28, 0.45),
}
PLASMA_BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (10, float("inf"))]
BIN_LABELS  = ["0-2", "2-4", "4-6", "6-8", "10+"]

H, W, D = VOL_SHAPE


def roi_mean(vol_np, roi):
    y0, y1, x0, x1, z0, z1 = roi
    return vol_np[int(y0*H):int(y1*H), int(x0*W):int(x1*W), int(z0*D):int(z1*D)].mean()


def roi_pearson(real_vol, gen_vol, roi):
    """Voxel-level Pearson within an ROI for a single subject."""
    y0, y1, x0, x1, z0, z1 = roi
    r = real_vol[int(y0*H):int(y1*H), int(x0*W):int(x1*W), int(z0*D):int(z1*D)].ravel()
    g = gen_vol[ int(y0*H):int(y1*H), int(x0*W):int(x1*W), int(z0*D):int(z1*D)].ravel()
    if r.std() < 1e-8 or g.std() < 1e-8:
        return float("nan")
    val, _ = pearsonr(r, g)
    return float(val)


def compute_metrics(real, gen):
    """Return dict of all 5 standard metrics for a pair of numpy volumes."""
    diff  = gen - real
    mse   = float(np.mean(diff ** 2))
    mae   = float(np.mean(np.abs(diff)))
    nrmse = float(np.sqrt(mse) / (real.max() - real.min() + 1e-8))
    ssim_score = float(ssim(real, gen, data_range=1.0))
    r, _  = pearsonr(real.ravel(), gen.ravel())
    return {"pearson": float(r), "nrmse": nrmse, "ssim": ssim_score, "mse": mse, "mae": mae}


def compute_roi_metrics(real_vals, gen_vals):
    """Metrics for 1-D arrays of per-subject ROI means (no SSIM — requires volumes)."""
    real, gen = np.array(real_vals), np.array(gen_vals)
    diff  = gen - real
    mse   = float(np.mean(diff ** 2))
    mae   = float(np.mean(np.abs(diff)))
    nrmse = float(np.sqrt(mse) / (real.max() - real.min() + 1e-8))
    r, _  = pearsonr(real, gen) if len(real) > 2 else (float("nan"), None)
    return {"pearson": float(r), "nrmse": nrmse, "mse": mse, "mae": mae}


def plot_loss_curves(ae_losses, diff_losses, save_path=None):
    if save_path is None:
        save_path = f"{FIGURES_DIR}/loss_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ae_losses)
    axes[0].set_title(f"AE Loss (100 epochs, final={ae_losses[-1]:.4f})")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(diff_losses)
    axes[1].set_title(f"Diffusion Loss (600 epochs, final={diff_losses[-1]:.4f})")
    axes[1].set_xlabel("Epoch")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Epoch 1 loss:   {diff_losses[0]:.4f}")
    if len(diff_losses) > 99:
        print(f"Epoch 100 loss: {diff_losses[99]:.4f}")
    print(f"Final loss:     {diff_losses[-1]:.4f}")


def plot_ae_reconstruction(ae, test_ds, device=DEVICE, save_path=None):
    if save_path is None:
        save_path = f"{FIGURES_DIR}/ae_recon.png"
    ae.eval()
    with torch.no_grad():
        pet, mri, _ = test_ds[0]
        recon, _, _ = ae(pet.unsqueeze(0).to(device))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    mid = 48
    axes[0].imshow(pet[0, :, :, mid].numpy(), cmap="hot")
    axes[0].set_title("Real PET")
    axes[1].imshow(recon[0, 0, :, :, mid].cpu().numpy(), cmap="hot")
    axes[1].set_title("AE Reconstruction")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()


def compare_generated_vs_real(real_vols, gen_vols, pearsons, nrmses, ssims, mses, maes,
                               mode, save_path=None):
    """Best / median / worst subject by SSIM + population mean real PET.

    Rows: Best, Median, Worst, Mean Real PET.
    Columns: Real PET, Generated PET, Abs Difference.
    Per-subject metrics are shown below the Generated PET panel for rows 0-2.
    Mean Real PET row shows mean real / mean generated / mean abs error.
    """
    if save_path is None:
        save_path = f"{FIGURES_DIR}/taugennet_comparison.png"

    ssim_arr   = np.array(ssims)
    best_idx   = int(np.argmax(ssim_arr))
    worst_idx  = int(np.argmin(ssim_arr))
    median_idx = int(np.argsort(ssim_arr)[len(ssim_arr) // 2])

    real_arr  = np.stack(real_vols)
    gen_arr   = np.stack(gen_vols)
    mean_real = real_arr.mean(axis=0)
    mean_gen  = gen_arr.mean(axis=0)
    mean_diff = np.abs(real_arr - gen_arr).mean(axis=0)

    def _metric_str(i):
        return (f"Pearson={pearsons[i]:.3f}  NRMSE={nrmses[i]:.3f}  SSIM={ssims[i]:.3f}"
                f"\nMSE={mses[i]:.4f}  MAE={maes[i]:.4f}")

    rows = [
        (real_vols[best_idx],   gen_vols[best_idx],   "Best",   _metric_str(best_idx)),
        (real_vols[median_idx], gen_vols[median_idx], "Median", _metric_str(median_idx)),
        (real_vols[worst_idx],  gen_vols[worst_idx],  "Worst",  _metric_str(worst_idx)),
        (mean_real,             mean_gen,             "Mean (all subjects)", None),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    col_titles = ["Real PET", "Generated PET", "Abs Difference"]

    for row_idx, (real, gen, row_label, metrics_str) in enumerate(rows):
        diff = mean_diff if metrics_str is None else np.abs(real - gen)
        mid  = real.shape[2] // 2   # axial midslice

        for col_idx, (slc, cmap, vmin, vmax) in enumerate([
            (real[:, :, mid], "hot",   0, 1),
            (gen[:, :, mid],  "hot",   0, 1),
            (diff[:, :, mid], "magma", 0, max(float(diff[:, :, mid].max()), 1e-6)),
        ]):
            ax = axes[row_idx, col_idx]
            ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight="bold")
            if col_idx == 1 and metrics_str is not None:
                ax.set_xlabel(metrics_str, fontsize=7.5, ha="center")
            ax.axis("off")

    plt.suptitle(
        f"Real vs Generated Tau PET — {mode}  |  "
        f"mean SSIM={np.mean(ssims):.3f} ± {np.std(ssims):.3f}  "
        f"mean MSE={np.mean(mses):.4f}  mean Pearson={np.mean(pearsons):.3f}"
        f"  [per-subject avg, voxel-level]",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison figure saved  -> {save_path}")


def run_steps_comparison(ae, unet, schedule, encode_cond, latent_std, test_ds,
                         device=DEVICE, save_path=None):
    if save_path is None:
        save_path = f"{FIGURES_DIR}/steps_comparison.png"
    ae.eval(); unet.eval()
    pet, mri, atrophy = test_ds[0]

    results = {}
    for n_steps in [1, 5, 10, 50]:
        gen = synthesize_tau_pet(mri.unsqueeze(0), atrophy, ae, unet, schedule,
                                 encode_cond, latent_std, device=device, n_steps=n_steps)
        results[n_steps] = gen.squeeze().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    mid = 48
    axes[0].imshow(pet[0, :, :, mid].numpy(), cmap="hot")
    axes[0].set_title("Real PET")
    for idx, n_steps in enumerate([1, 5, 10, 50]):
        axes[idx+1].imshow(results[n_steps][:, :, mid], cmap="hot")
        axes[idx+1].set_title(f"{n_steps} steps")
    for ax in axes: ax.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()


def run_mri_ablation_visual(ae, unet, schedule, encode_cond, latent_std, test_ds,
                             device=DEVICE, save_path=None):
    if save_path is None:
        save_path = f"{FIGURES_DIR}/mri_ablation.png"
    """Generate with real MRI vs zeroed MRI to visualise structural guidance."""
    pet, mri, atrophy = test_ds[0]

    gen_normal = synthesize_tau_pet(mri.unsqueeze(0), atrophy, ae, unet, schedule,
                                    encode_cond, latent_std, device=device,
                                    n_steps=500, sampler='ddpm').squeeze().numpy()

    mri_zero   = torch.zeros_like(mri)
    gen_no_mri = synthesize_no_mri(mri_zero.unsqueeze(0), atrophy, ae, unet, schedule,
                                   encode_cond, latent_std, device=device,
                                   n_steps=500).squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    mid = 48
    axes[0].imshow(pet[0, :, :, mid].numpy(), cmap="hot")
    axes[0].set_title("Real PET")
    axes[1].imshow(gen_normal[:, :, mid], cmap="hot")
    axes[1].set_title("Generated (real MRI)")
    axes[2].imshow(gen_no_mri[:, :, mid], cmap="hot")
    axes[2].set_title("Generated (zero MRI)")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()

    diff = np.mean(np.abs(gen_normal - gen_no_mri))
    print(f"Mean difference with/without MRI: {diff:.4f}")


def run_multi_subject_metrics(ae, unet, schedule, encode_cond, latent_std, test_ds,
                               device=DEVICE, n_steps=500):
    """Compute per-subject metrics over all test subjects; also returns volumes for figures."""
    pearsons, nrmses, ssims, mses, maes = [], [], [], [], []
    real_vols, gen_vols = [], []
    for i in range(len(test_ds)):
        pet, mri, atrophy = test_ds[i]
        real = pet.squeeze().numpy()
        gen  = synthesize_tau_pet(mri.unsqueeze(0), atrophy, ae, unet, schedule,
                                  encode_cond, latent_std, device=device,
                                  n_steps=n_steps, sampler='ddpm').squeeze().numpy()
        real_vols.append(real)
        gen_vols.append(gen)
        m = compute_metrics(real, gen)
        pearsons.append(m["pearson"])
        nrmses.append(m["nrmse"])
        ssims.append(m["ssim"])
        mses.append(m["mse"])
        maes.append(m["mae"])

    print(f"Mean Pearson: {np.mean(pearsons):.4f} ± {np.std(pearsons):.4f}")
    print(f"Mean NRMSE:   {np.mean(nrmses):.4f} ± {np.std(nrmses):.4f}")
    print(f"Mean SSIM:    {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
    print(f"Mean MSE:     {np.mean(mses):.4f} ± {np.std(mses):.4f}")
    print(f"Mean MAE:     {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    return pearsons, nrmses, ssims, mses, maes, real_vols, gen_vols


def plot_population_comparison(real_vols, gen_vols, mode, save_path):
    """Mean real / mean generated / mean abs error across all test subjects, 3 views."""
    real_arr  = np.stack(real_vols)                          # (N, H, W, D)
    gen_arr   = np.stack(gen_vols)
    mean_real = real_arr.mean(axis=0)
    mean_gen  = gen_arr.mean(axis=0)
    mean_err  = np.abs(real_arr - gen_arr).mean(axis=0)

    views = {"Axial": 2, "Sagittal": 0, "Coronal": 1}
    fig, axes = plt.subplots(len(views), 3, figsize=(12, 3 * len(views)))

    for row_idx, (view_name, axis_dim) in enumerate(views.items()):
        mid = mean_real.shape[axis_dim] // 2
        r_slc = np.take(mean_real, mid, axis=axis_dim)
        g_slc = np.take(mean_gen,  mid, axis=axis_dim)
        e_slc = np.take(mean_err,  mid, axis=axis_dim)

        for col_idx, (slc, title, cmap, vmin, vmax) in enumerate([
            (r_slc, "Mean Real PET",   "hot",   0, 1),
            (g_slc, "Mean Generated",  "hot",   0, 1),
            (e_slc, "Mean Abs Error",  "magma", 0, max(float(e_slc.max()), 1e-6)),
        ]):
            ax = axes[row_idx, col_idx]
            ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(view_name, fontsize=10)
            ax.axis("off")

    plt.suptitle(f"Population Mean — {mode} (n={len(real_vols)} subjects)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Population comparison saved -> {save_path}")


def plot_subject_comparison(real_vols, gen_vols, nrmses, mode, save_path):
    """Best / median / worst subject by NRMSE, axial midslice, 3 columns."""
    nrmse_arr  = np.array(nrmses)
    best_idx   = int(np.argmin(nrmse_arr))
    worst_idx  = int(np.argmax(nrmse_arr))
    median_idx = int(np.argsort(nrmse_arr)[len(nrmse_arr) // 2])

    subjects = [
        (best_idx,   f"Best   (NRMSE={nrmses[best_idx]:.3f})"),
        (median_idx, f"Median (NRMSE={nrmses[median_idx]:.3f})"),
        (worst_idx,  f"Worst  (NRMSE={nrmses[worst_idx]:.3f})"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))

    for row_idx, (idx, row_label) in enumerate(subjects):
        real = real_vols[idx]
        gen  = gen_vols[idx]
        diff = np.abs(real - gen)
        mid  = real.shape[2] // 2   # axial midslice

        for col_idx, (slc, title, cmap, vmin, vmax) in enumerate([
            (real[:, :, mid], "Real PET",       "hot",   0, 1),
            (gen[:, :, mid],  "Generated PET",  "hot",   0, 1),
            (diff[:, :, mid], "Abs Difference", "magma", 0, max(float(diff.max()), 1e-6)),
        ]):
            ax = axes[row_idx, col_idx]
            ax.imshow(slc, cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=9)
            ax.axis("off")

    plt.suptitle(f"Best / Median / Worst Subject — {mode}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Subject comparison saved  -> {save_path}")


def run_roi_mse_eval(ae, unet, schedule, encode_cond, latent_std, test_loader,
                     device=DEVICE, n_steps=20):
    """Region-wise metrics across all test subjects (reproduces Table II structure).

    Pearson: mean of per-subject voxel-level Pearson within each ROI.
    MSE / MAE / NRMSE: pooled from per-subject ROI means.

    NOTE: The original notebook stratified by scalar p-tau217 plasma bins, but
    the dataset was later updated to return 86-dim atrophy vectors instead of a
    scalar. Binning is therefore skipped here; all subjects are pooled.
    """
    rec_real  = {r: [] for r in ROI_DEFS}
    rec_gen   = {r: [] for r in ROI_DEFS}
    rec_pears = {r: [] for r in ROI_DEFS}

    print("Running evaluation on test set...")
    for pet, mri, ptau in tqdm(test_loader):
        for i in range(pet.shape[0]):
            mri_i  = mri[i:i+1]
            pet_np = pet[i].squeeze().numpy()
            gen_np = synthesize_tau_pet(mri_i, ptau[i], ae, unet, schedule,
                                        encode_cond, latent_std, device=device,
                                        n_steps=n_steps).squeeze().numpy()
            for rname, roi in ROI_DEFS.items():
                rec_real[rname].append(roi_mean(pet_np, roi))
                rec_gen[rname].append(roi_mean(gen_np, roi))
                rec_pears[rname].append(roi_pearson(pet_np, gen_np, roi))

    metric_names = ["pearson", "nrmse", "mse", "mae"]
    tables = {k: pd.DataFrame(index=["All subjects"], columns=list(ROI_DEFS.keys()), dtype=float)
              for k in metric_names}
    for rname in ROI_DEFS:
        r, g = rec_real[rname], rec_gen[rname]
        if not r:
            continue
        m = compute_roi_metrics(r, g)
        for k in ["nrmse", "mse", "mae"]:
            tables[k].loc["All subjects", rname] = round(m[k], 6)
        valid = [p for p in rec_pears[rname] if not np.isnan(p)]
        tables["pearson"].loc["All subjects", rname] = round(float(np.mean(valid)), 6) if valid else float("nan")

    _label = {"pearson": "per-subject avg, voxel-level",
              "nrmse":   "pooled ROI means",
              "mse":     "pooled ROI means",
              "mae":     "pooled ROI means"}
    for k in metric_names:
        print(f"\nRegion-wise {k.upper()} [{_label[k]}]:")
        print(tables[k].to_string())
    return tables


def run_ablation_study(ae, unet, schedule, encode_cond, latent_std, test_loader,
                       device=DEVICE, n_steps=20):
    """MRI+Atrophy vs Atrophy-only ablation (reproduces Table I).

    Pearson is the mean of per-subject voxel-level Pearson within each ROI.
    MSE/MAE/NRMSE are computed on pooled per-subject ROI means.
    """
    means   = {k: {r: [] for r in ROI_DEFS} for k in ["Real", "MRI+Atrophy", "Atrophy-only"]}
    pearsons = {"MRI+Atrophy": {r: [] for r in ROI_DEFS},
                "Atrophy-only": {r: [] for r in ROI_DEFS}}

    print("Running ablation study...")
    for pet, mri, ptau in tqdm(test_loader):
        for i in range(pet.shape[0]):
            mri_i     = mri[i:i+1]
            pet_np    = pet[i].squeeze().numpy()
            gen_full  = synthesize_tau_pet(mri_i, ptau[i], ae, unet, schedule,
                                           encode_cond, latent_std, device=device,
                                           n_steps=n_steps).squeeze().numpy()
            gen_ablat = synthesize_no_mri(mri_i, ptau[i], ae, unet, schedule,
                                          encode_cond, latent_std, device=device,
                                          n_steps=n_steps).squeeze().numpy()
            for rname, roi in ROI_DEFS.items():
                means["Real"][rname].append(roi_mean(pet_np, roi))
                means["MRI+Atrophy"][rname].append(roi_mean(gen_full, roi))
                means["Atrophy-only"][rname].append(roi_mean(gen_ablat, roi))
                pearsons["MRI+Atrophy"][rname].append(roi_pearson(pet_np, gen_full, roi))
                pearsons["Atrophy-only"][rname].append(roi_pearson(pet_np, gen_ablat, roi))

    metric_names = ["pearson", "nrmse", "mse", "mae"]
    tables = {k: pd.DataFrame(index=["Atrophy-only", "MRI+Atrophy"],
                              columns=list(ROI_DEFS.keys()), dtype=float)
              for k in metric_names}
    for cond in ["Atrophy-only", "MRI+Atrophy"]:
        for rname in ROI_DEFS:
            r, g = means["Real"][rname], means[cond][rname]
            if not r:
                continue
            m = compute_roi_metrics(r, g)
            for k in ["nrmse", "mse", "mae"]:
                tables[k].loc[cond, rname] = round(m[k], 6)
            # per-subject average voxel-level Pearson
            valid = [p for p in pearsons[cond][rname] if not np.isnan(p)]
            tables["pearson"].loc[cond, rname] = round(float(np.mean(valid)), 6) if valid else float("nan")

    _label = {"pearson": "per-subject avg, voxel-level",
              "nrmse":   "pooled ROI means",
              "mse":     "pooled ROI means",
              "mae":     "pooled ROI means"}
    for k in metric_names:
        print(f"\nAblation {k.upper()} [{_label[k]}] – cf. Table I:")
        print(tables[k].to_string())
    return tables


def plot_metrics_boxplot(pearsons, nrmses, ssims, mses, maes, mode, save_path):
    """Boxplot of all 5 per-subject metrics with individual points and summary stats."""
    metrics = {
        "Pearson": pearsons,
        "NRMSE":   nrmses,
        "SSIM":    ssims,
        "MSE":     mses,
        "MAE":     maes,
    }
    fig, axes = plt.subplots(1, 5, figsize=(16, 5))
    fig.suptitle(f"Per-subject metrics — {mode} mode  (n={len(pearsons)})",
                 fontsize=13, fontweight="bold")

    rng = np.random.default_rng(0)
    for ax, (name, vals) in zip(axes, metrics.items()):
        vals_arr = np.array(vals)
        bp = ax.boxplot(vals_arr, patch_artist=True, widths=0.5,
                        boxprops=dict(facecolor="#aec6e8", color="#2c5f8a"),
                        medianprops=dict(color="#d62728", linewidth=2),
                        whiskerprops=dict(color="#2c5f8a"),
                        capprops=dict(color="#2c5f8a"),
                        flierprops=dict(marker="o", markersize=4, color="#999999"))

        # jittered individual points
        jitter = rng.uniform(-0.12, 0.12, size=len(vals_arr))
        ax.scatter(1 + jitter, vals_arr, alpha=0.55, s=18, color="#1f77b4", zorder=3)

        median = float(np.median(vals_arr))
        mean   = float(np.mean(vals_arr))
        std    = float(np.std(vals_arr))
        ax.axhline(mean, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"mean")

        # annotate median and mean ± std below the plot
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_xlabel(
            f"median {median:.3f}\nmean {mean:.3f} ± {std:.3f}",
            fontsize=8.5
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_multi_subject_ssim(ssims, mode, save_path):
    """Bar chart of per-subject SSIM across the test set."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(ssims)), ssims, color="steelblue")
    ax.axhline(np.mean(ssims), color="red", linestyle="--",
               label=f"Mean = {np.mean(ssims):.3f} ± {np.std(ssims):.3f}")
    ax.set_xlabel("Test subject index")
    ax.set_ylabel("SSIM")
    ax.set_title(f"Per-subject SSIM — {mode} mode")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roi_heatmap(mse_table, mode, save_path):
    """Heatmap of region-wise MSE table."""
    fig, ax = plt.subplots(figsize=(11, 2))
    data = mse_table.values.astype(float)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(mse_table.columns)))
    ax.set_xticklabels(mse_table.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(mse_table.index)))
    ax.set_yticklabels(mse_table.index, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.4f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="MSE")
    ax.set_title(f"Region-wise MSE [pooled ROI means] — {mode} mode")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ablation_roi_comparison(ablation_tables, mode, save_path):
    """Grouped bar chart: MRI+Atrophy vs Atrophy-only per ROI for MSE and Pearson."""
    rois = list(ROI_DEFS.keys())
    x    = np.arange(len(rois))
    w    = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel, title in [
        (axes[0], "mse",     "MSE",     "Region-wise MSE\n[pooled ROI means]"),
        (axes[1], "pearson", "Pearson", "Region-wise Pearson\n[per-subject avg, voxel-level]"),
    ]:
        tbl = ablation_tables[metric]
        mri_vals   = [float(tbl.loc["MRI+Atrophy",   r]) for r in rois]
        ablat_vals = [float(tbl.loc["Atrophy-only",   r]) for r in rois]

        ax.bar(x - w/2, mri_vals,   w, label="MRI+Atrophy",   color="#4c72b0")
        ax.bar(x + w/2, ablat_vals, w, label="Atrophy-only",  color="#dd8452")
        ax.set_xticks(x)
        ax.set_xticklabels(rois, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle(f"MRI Ablation — Per-ROI Metrics ({mode})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation ROI comparison saved -> {save_path}")


# ── checkpoint-dir → figures subdirectory mapping ─────────────────────────────
_DIR_TO_SUBPATH = {
    "atrophy":    os.path.join("atrophy",  "72split"),
    "ptau217":    os.path.join("ptau217",  "72split"),
    "atrophy_v2": os.path.join("atrophy",  "80split"),
    "ptau217_v2": os.path.join("ptau217",  "80split"),
    "combined":   "atrophy_ptau217",
}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["atrophy", "ptau217", "combined"], default="atrophy")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Override default checkpoint directory.")
    p.add_argument("--figures-dir", type=str, default=None,
                   help="Override figures subdirectory (relative to results/figures/). "
                        "Auto-detected from --checkpoint-dir if not set.")
    p.add_argument("--use-mask", action="store_true", default=False,
                   help="Apply gray-matter brain mask (T1_seg label==2) to MRI and PET volumes")
    p.add_argument("--arch", choices=["silu", "relu"], default="silu",
                   help="Model architecture variant: 'silu' (default) or 'relu'")
    args = p.parse_args()

    COND_MODE = args.mode

    # ── checkpoint path ────────────────────────────────────────────────────────
    if args.checkpoint_dir:
        CKPT_PATH = os.path.join(args.checkpoint_dir, f"diff_{COND_MODE}.pt")
    else:
        CKPT_PATH = f"results/checkpoints/{COND_MODE}/diff_{COND_MODE}.pt"

    # ── figures output directory ───────────────────────────────────────────────
    if args.figures_dir:
        fig_subpath = args.figures_dir
    elif args.checkpoint_dir:
        key = os.path.basename(args.checkpoint_dir.rstrip("/"))
        fig_subpath = _DIR_TO_SUBPATH.get(key, key)
    else:
        fig_subpath = _DIR_TO_SUBPATH.get(COND_MODE, COND_MODE)

    FIG_DIR = os.path.join(FIGURES_DIR, fig_subpath)
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Figures → {FIG_DIR}")

    # ── dataset ───────────────────────────────────────────────────────────────
    if args.arch == "relu":
        from src import dataset_v2_relu as _dataset_v2_relu
        from src import dataset_combined_relu as _dataset_combined_relu
        if args.mode == "combined":
            train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
                _dataset_combined_relu.build_dataloaders(mode=COND_MODE, use_dk_mask=args.use_mask)
        else:
            train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
                _dataset_v2_relu.build_dataloaders(mode=COND_MODE, use_dk_mask=args.use_mask)
    else:
        if args.mode == "combined":
            train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
                _dataset_combined.build_dataloaders(mode=COND_MODE, use_dk_mask=args.use_mask)
        else:
            train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = \
                _dataset_final.build_dataloaders(mode=COND_MODE, use_dk_mask=args.use_mask)

    ae, unet, conditioner, latent_std, diff_losses = load_models(CKPT_PATH, COND_MODE, arch=args.arch)
    encode_cond = conditioner.encode
    schedule    = DiffusionSchedule()
    print(f"latent_std per channel: {latent_std.squeeze().tolist()}")

    # Loss curve
    print(f"Final diffusion loss: {diff_losses[-1]:.6f}")
    plt.figure(figsize=(7, 3))
    plt.plot(diff_losses); plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title(f"Diffusion Loss — {COND_MODE} mode"); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/diff_loss_{COND_MODE}.png", dpi=150)
    plt.close()

    # AE reconstruction quality
    plot_ae_reconstruction(ae, test_ds,
                           save_path=f"{FIG_DIR}/ae_recon.png")

    # Denoising steps comparison
    run_steps_comparison(ae, unet, schedule, encode_cond, latent_std, test_ds,
                         save_path=f"{FIG_DIR}/steps_{COND_MODE}.png")

    # MRI ablation visualisation
    run_mri_ablation_visual(ae, unet, schedule, encode_cond, latent_std, test_ds,
                            save_path=f"{FIG_DIR}/mri_ablation_{COND_MODE}.png")

    # Multi-subject metrics (computes all vols once; reused by several plots below)
    pearsons, nrmses, ssims, mses, maes, real_vols, gen_vols = run_multi_subject_metrics(
        ae, unet, schedule, encode_cond, latent_std, test_ds)
    plot_multi_subject_ssim(ssims, COND_MODE,
                            save_path=f"{FIG_DIR}/multi_subject_ssim_{COND_MODE}.png")
    plot_metrics_boxplot(pearsons, nrmses, ssims, mses, maes, COND_MODE,
                         save_path=f"{FIG_DIR}/metrics_boxplot_{COND_MODE}.png")
    plot_population_comparison(real_vols, gen_vols, COND_MODE,
                               save_path=f"{FIG_DIR}/population_comparison_{COND_MODE}.png")
    plot_subject_comparison(real_vols, gen_vols, nrmses, COND_MODE,
                            save_path=f"{FIG_DIR}/subject_comparison_{COND_MODE}.png")

    # Best/median/worst + mean real PET comparison (uses pre-computed vols)
    compare_generated_vs_real(real_vols, gen_vols, pearsons, nrmses, ssims, mses, maes,
                              COND_MODE, save_path=f"{FIG_DIR}/comparison_{COND_MODE}.png")

    # ROI metrics tables + heatmap (MSE)
    roi_tables = run_roi_mse_eval(ae, unet, schedule, encode_cond, latent_std, test_loader)
    plot_roi_heatmap(roi_tables["mse"], COND_MODE,
                     save_path=f"{FIG_DIR}/roi_heatmap_{COND_MODE}.png")

    # Ablation study (Table I) + per-ROI bar chart
    ablation_tables = run_ablation_study(ae, unet, schedule, encode_cond, latent_std, test_loader)
    plot_ablation_roi_comparison(ablation_tables, COND_MODE,
                                 save_path=f"{FIG_DIR}/ablation_roi_{COND_MODE}.png")
