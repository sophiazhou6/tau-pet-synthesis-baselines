#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
evaluate_cfg.py — Evaluate a CFG-trained TauGenNet checkpoint.

Identical metric/figure pipeline to evaluate_final.py, but uses two U-Net
forward passes per denoising step:
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

Generated .npy files are saved to results/generated/cfg/<mode>/ by default.
Existing .npy files are NEVER overwritten.

Usage
-----
python scripts/evaluate_cfg.py --mode atrophy
python scripts/evaluate_cfg.py --mode atrophy --guidance-scale 5.0
python scripts/evaluate_cfg.py --mode atrophy --use-cached   # skip inference
"""

import os
import sys
import argparse

import numpy as np
import torch
from tqdm import tqdm

# ── Re-use everything from evaluate_final except generate_all ─────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_final import (
    compute_wholebrain_metrics,
    compute_crosssubject_voxelwise_r,
    compute_crosssubject_roi_r,
    compute_roi_metrics,
    compute_regional_overall,
    compute_regional_overall_normalized,
    compute_ablation_metrics,
    compute_plasma_tables,
    compute_suvr_range,
    load_cached_generations,
    plot_subject_comparison,
    plot_population_comparison,
    plot_wholebrain_boxplot,
    plot_roi_metrics_boxplot,
    plot_crosssubject_r_map,
    plot_crosssubject_roi_r,
    plot_regional_overall_heatmap,
    plot_plasma_heatmap,
    save_tables_to_disk,
    _pet_norms,
    _plasma_value,
)

from src.config import DEVICE, FIGURES_DIR
from src import dataset_final as _dataset_v2
from src import dataset_combined as _dataset_combined
from src.dataset_final import unnormalize as _unnormalize
from src.diffusion import DiffusionSchedule
from src.inference import load_models, _prepare_cond


# ── CFG inference ─────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_all_cfg(ae, unet, schedule, encode_cond, latent_std, test_ds,
                     cond_mode, guidance_scale=3.0, n_steps=500, save_dir=None):
    """Run CFG-guided DDPM inference on every test subject.

    Two U-Net passes per step:
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    Existing subject_*.npy files in save_dir are skipped (no overwrite).
    Returns the same tuple as evaluate_final.generate_all().
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    step = 500 // n_steps
    ts   = list(reversed(range(0, 500, step)))

    real_norm, gen_masked, gen_raw = [], [], []
    real_suvr, gen_suvr = [], []
    plasma_vals = [] if cond_mode in ("ptau217", "combined") else None

    for i in tqdm(range(len(test_ds)), desc="CFG inference"):
        pet, mri, cond = test_ds[i]
        real = pet.squeeze().numpy()

        gen_path = os.path.join(save_dir, f"subject_{i:03d}.npy") if save_dir else None
        if gen_path and os.path.exists(gen_path):
            gen = np.load(gen_path).astype(np.float32)
            print(f"  subject_{i:03d}: loaded from cache (not overwritten)")
        else:
            mri_t  = mri.unsqueeze(0).to(DEVICE)
            cond_t = _prepare_cond(cond, DEVICE)
            zm     = ae.encode_mean(mri_t) / latent_std
            ctx      = encode_cond(cond_t)
            ctx_null = torch.zeros_like(ctx)
            zt = torch.randn_like(zm)

            for t_idx in tqdm(ts, desc=f"  s{i:03d}", leave=False):
                t_batch    = torch.full((1,), t_idx, device=DEVICE, dtype=torch.long)
                ht         = torch.cat([zt, zm], dim=1)
                eps_cond   = unet(ht, t_batch, ctx)
                eps_uncond = unet(ht, t_batch, ctx_null)
                eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                zt = schedule.p_sample_from_eps(zt, t_idx, eps_guided)

            gen = ae.decode(zt * latent_std).squeeze().cpu().numpy()

            if gen_path:
                np.save(gen_path, gen)

        real_norm.append(real)
        gen_raw.append(gen)
        gen_masked.append(gen * (real > 0))

        pmin, pmax = _pet_norms(test_ds, i)
        real_suvr.append(_unnormalize(real, pmin, pmax))
        gen_suvr.append(_unnormalize(gen,  pmin, pmax))

        if plasma_vals is not None:
            plasma_vals.append(_plasma_value(cond))

    if save_dir:
        print(f"Generated volumes → {save_dir}")
    return real_norm, gen_masked, gen_raw, real_suvr, gen_suvr, plasma_vals


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="TauGenNet CFG evaluation")
    p.add_argument("--mode",           choices=["atrophy", "ptau217", "combined"], required=True)
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Directory containing diff_{mode}_best.pt (default: results/checkpoints/cfg/<mode>)")
    p.add_argument("--figures-dir",    type=str, default=None,
                   help="Output figures directory (default: results/figures/cfg/<mode>)")
    p.add_argument("--records-dir",    type=str, default="results/records/cfg",
                   help="Output CSV/markdown directory (default: results/records/cfg)")
    p.add_argument("--generated-dir",  type=str, default=None,
                   help="Directory for generated .npy files (default: results/generated/cfg/<mode>)")
    p.add_argument("--use-mask",       action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-steps",        type=int, default=500)
    p.add_argument("--guidance-scale", type=float, default=None,
                   help="CFG guidance scale (default: read from checkpoint, fallback 3.0)")
    p.add_argument("--unet-channels",  type=str, default="128,256,512")
    p.add_argument("--n-transformer",  type=int, default=1)
    p.add_argument("--use-best",       action="store_true", default=True,
                   help="Load diff_{mode}_best.pt (default: on)")
    p.add_argument("--use-cached",     action="store_true", default=False,
                   help="Load existing .npy files instead of running inference")
    p.add_argument("--skip-ablation",  action="store_true", default=True)
    args = p.parse_args()

    mode = args.mode
    ckpt_dir    = args.checkpoint_dir or f"results/checkpoints/cfg/{mode}"
    fig_dir     = args.figures_dir    or os.path.join(FIGURES_DIR, "cfg", mode)
    gen_dir     = args.generated_dir  or f"results/generated/cfg/{mode}"
    rec_dir     = args.records_dir

    suffix   = "_best" if args.use_best else ""
    ckpt_path = os.path.join(ckpt_dir, f"diff_{mode}{suffix}.pt")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)
    print(f"Mode:        {mode}")
    print(f"Checkpoint:  {ckpt_path}")
    print(f"Figures  →   {fig_dir}")
    print(f"Records  →   {rec_dir}")
    print(f"Generated → {gen_dir}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    # CFG checkpoints are legacy-trained -> force the legacy split (avoid mentor-split leakage).
    if mode == "combined":
        _, _, test_ds, *_ = _dataset_combined.build_dataloaders(
            mode=mode, use_dk_mask=args.use_mask, use_mentor_split=False)
    else:
        _, _, test_ds, *_ = _dataset_v2.build_dataloaders(
            mode=mode, use_dk_mask=args.use_mask, use_mentor_split=False)
    print(f"Test subjects: {len(test_ds)}")

    # ── Inference ─────────────────────────────────────────────────────────────
    if args.use_cached:
        print(f"Loading cached generations from: {gen_dir}")
        real_norm, gen_masked, gen_raw, real_suvr, gen_suvr, plasma_vals = \
            load_cached_generations(test_ds, mode, gen_dir)
    else:
        ch_list = tuple(int(x) for x in args.unet_channels.split(","))
        ae, unet, conditioner, latent_std, _ = load_models(
            ckpt_path, mode, arch="silu",
            ch_list=ch_list, n_transformer=args.n_transformer)
        schedule = DiffusionSchedule()

        # Read guidance_scale from checkpoint if not overridden
        guidance_scale = args.guidance_scale
        if guidance_scale is None:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            guidance_scale = ckpt.get("guidance_scale", 3.0)
        print(f"Guidance scale: {guidance_scale}")

        real_norm, gen_masked, gen_raw, real_suvr, gen_suvr, plasma_vals = \
            generate_all_cfg(
                ae, unet, schedule, conditioner.encode, latent_std, test_ds,
                cond_mode=mode, guidance_scale=guidance_scale,
                n_steps=args.n_steps, save_dir=gen_dir,
            )

    # ── Metrics & figures (identical to evaluate_final) ───────────────────────
    pet_vmin, pet_vmax, _, err_vmax = compute_suvr_range(real_suvr, gen_suvr)
    pearsons, nrmses, ssims, mses, maes = compute_wholebrain_metrics(real_norm, gen_masked)

    plot_subject_comparison(
        real_suvr, gen_suvr, pearsons, nrmses, ssims, mses, maes,
        mode, pet_vmin, pet_vmax, err_vmax,
        save_path=os.path.join(fig_dir, f"subject_comparison_{mode}.png"))
    plot_population_comparison(
        real_suvr, gen_suvr, mode, pet_vmin, pet_vmax, err_vmax,
        save_path=os.path.join(fig_dir, f"population_comparison_{mode}.png"))
    plot_wholebrain_boxplot(
        pearsons, nrmses, ssims, mses, maes, mode,
        save_path=os.path.join(fig_dir, f"wholebrain_boxplot_{mode}.png"))

    cs_brain_mask = (np.stack(real_norm, axis=0) > 0).any(axis=0)
    cs_r_map  = compute_crosssubject_voxelwise_r(real_norm, gen_masked,
                                                  brain_mask=cs_brain_mask)
    cs_roi_r  = compute_crosssubject_roi_r(cs_r_map)
    cs_mean_r = float(cs_r_map[cs_brain_mask].mean())
    plot_crosssubject_r_map(cs_r_map, real_norm, mode,
        save_path=os.path.join(fig_dir, f"crosssubject_r_map_{mode}.png"))
    plot_crosssubject_roi_r(cs_roi_r, mode,
        save_path=os.path.join(fig_dir, f"crosssubject_roi_r_{mode}.png"))

    gen_suvr_masked  = [g * (r > 0) for g, r in zip(gen_suvr, real_norm)]
    cs_r_map_suvr    = compute_crosssubject_voxelwise_r(
        real_suvr, gen_suvr_masked, brain_mask=cs_brain_mask,
        label=" [SUVR / burden-included]")
    cs_roi_r_suvr    = compute_crosssubject_roi_r(cs_r_map_suvr)
    cs_mean_r_suvr   = float(cs_r_map_suvr[cs_brain_mask].mean())
    plot_crosssubject_r_map(cs_r_map_suvr, real_norm, f"{mode} (SUVR)",
        save_path=os.path.join(fig_dir, f"crosssubject_r_map_{mode}_suvr.png"))
    plot_crosssubject_roi_r(cs_roi_r_suvr, f"{mode} (SUVR)",
        save_path=os.path.join(fig_dir, f"crosssubject_roi_r_{mode}_suvr.png"))

    roi_tables, roi_raw = compute_roi_metrics(real_norm, gen_masked)
    plot_roi_metrics_boxplot(roi_raw, mode,
        save_path=os.path.join(fig_dir, f"roi_metrics_boxplot_{mode}.png"))

    regional_tables, Rreal, Rgen = compute_regional_overall(real_suvr, gen_suvr)
    regional_tables["ssim"] = roi_tables["ssim"]
    plot_regional_overall_heatmap(regional_tables, mode, fig_dir)
    regional_norm_table = compute_regional_overall_normalized(real_norm, gen_raw)

    crosssubject_summary = {
        "normalized": {"_mean": cs_mean_r,      **cs_roi_r},
        "SUVR":       {"_mean": cs_mean_r_suvr, **cs_roi_r_suvr},
    }

    plasma_tables, plasma_counts = None, None
    if plasma_vals is not None:
        plasma_arr = np.asarray(plasma_vals)
        print(f"\nPlasma p-tau217 range: {plasma_arr.min():.2f} – {plasma_arr.max():.2f}")
        plasma_tables, plasma_counts = compute_plasma_tables(
            real_suvr, gen_suvr, plasma_vals, Rreal=Rreal, Rgen=Rgen)
        for metric in ("nrmse", "mse", "mae", "pearson"):
            plot_plasma_heatmap(plasma_tables[metric], metric, mode,
                save_path=os.path.join(fig_dir, f"plasma_{metric}_{mode}.png"))

    save_tables_to_disk(roi_tables, None, plasma_tables, plasma_counts,
                        regional_tables, mode, rec_dir,
                        regional_norm_table=regional_norm_table,
                        crosssubject_summary=crosssubject_summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
