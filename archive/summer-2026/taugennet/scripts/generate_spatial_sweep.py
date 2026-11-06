#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""Generate tau PET for spatial atrophy-map model.

Saves subject_000.npy ... subject_N.npy to --generated-dir.
Afterwards run:
    evaluate_final.py --use-cached --generated-dir <dir> --mode atrophy ...
for full metrics and figures.
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import DEVICE, T_STEPS, LATENT_CH
from src import dataset_spatial as _dataset
from src.models_spatial_sweep   import Autoencoder3D, DenoisingUNet3D
from src.diffusion         import DiffusionSchedule
from src.conditioning      import (AtrophyConditioner, NullConditioner, PTau217Conditioner,
                                   PTau217MLPConditioner)


@torch.no_grad()
def _ddpm_loop(ae, unet, conditioner, schedule, latent_std,
               mri, atrophy_vec, atrophy_map, device, n_steps=500, tissue=None,
               guidance_scale=1.0):
    """Single-sample DDPM for spatial model: ht = cat(zt, zm, cond_latent),
    where cond_latent is the atrophy map (+ tissue one-hot if provided).

    guidance_scale > 1 -> classifier-free guidance: a second UNet pass with the atrophy map AND
    cross-attention context zeroed gives eps_uncond, then
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond).
    Requires a checkpoint trained with --cfg-prob>0, else the uncond pass is meaningless."""
    mri_b = mri.unsqueeze(0).to(device)
    avec  = atrophy_vec.unsqueeze(0).to(device)
    amap  = atrophy_map.unsqueeze(0).to(device)  # (1,1,H,W,D)

    zm  = ae.encode_mean(mri_b) / latent_std
    ctx = conditioner.encode(avec)
    zt  = torch.randn_like(zm)

    am_latent = F.avg_pool3d(amap, kernel_size=8)  # (1,1,12,14,12)
    if tissue is not None:
        tissue_latent = F.avg_pool3d(tissue.unsqueeze(0).to(device), kernel_size=8)  # (1,3,...)
        am_latent     = torch.cat([am_latent, tissue_latent], dim=1)                 # (1,4,...)

    use_cfg  = guidance_scale is not None and abs(guidance_scale - 1.0) > 1e-6
    am_zero  = torch.zeros_like(am_latent)   # unconditional: atrophy map dropped
    ctx_zero = torch.zeros_like(ctx)         # unconditional: cross-attention context dropped

    step = T_STEPS // n_steps
    ts   = list(reversed(range(0, T_STEPS, step)))
    for i, t_idx in enumerate(ts):
        t_batch = torch.full((1,), t_idx, device=device, dtype=torch.long)
        ht      = torch.cat([zt, zm, am_latent], dim=1)
        eps_hat = unet(ht, t_batch, ctx, cond_map=am_latent)  # cond_map ignored unless use_spade
        if use_cfg:
            ht_u    = torch.cat([zt, zm, am_zero], dim=1)      # keep z_m; drop the conditioning
            eps_un  = unet(ht_u, t_batch, ctx_zero, cond_map=am_zero)
            eps_hat = eps_un + guidance_scale * (eps_hat - eps_un)

        sqrt_ab      = schedule.sqrt_ab[t_idx]
        sqrt_one_m_ab = schedule.sqrt_one_m_ab[t_idx]
        z0_hat = (zt - sqrt_one_m_ab * eps_hat) / sqrt_ab

        if t_idx == 0:
            zt = z0_hat
        else:
            beta_t   = schedule.betas[t_idx]
            alpha_t  = schedule.alphas[t_idx]
            ab_prev  = schedule.alpha_bar_prev[t_idx]
            coef1    = ab_prev.sqrt() * beta_t / (1 - schedule.alpha_bar[t_idx])
            coef2    = alpha_t.sqrt() * (1 - ab_prev) / (1 - schedule.alpha_bar[t_idx])
            mean     = coef1 * z0_hat + coef2 * zt
            zt       = mean + schedule.post_var[t_idx].sqrt() * torch.randn_like(zt)

    return ae.decode(zt * latent_std).squeeze().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True,
                   help="Path to spatial diffusion checkpoint .pt")
    p.add_argument("--generated-dir", required=True,
                   help="Directory to save subject_NNN.npy files")
    p.add_argument("--latent-ch",     type=int, default=LATENT_CH)
    p.add_argument("--unet-channels", type=str, default="128,256,512")
    p.add_argument("--n-transformer", type=int, default=1)
    p.add_argument("--n-steps",       type=int, default=500)
    p.add_argument("--noise-schedule", choices=["linear", "cosine"], default=None,
                   help="DDPM beta schedule for the reverse process. MUST match the "
                        "schedule the checkpoint was trained with; a mismatch silently "
                        "degrades samples. Default: read 'noise_schedule' from the "
                        "checkpoint (falling back to 'linear' for older checkpoints).")
    p.add_argument("--use-mask",      action="store_true", default=True)
    p.add_argument("--phase6-manifest", type=str, default=None,
                   help="Pooled AD+MCI+CN manifest; phase6 test set via dataset_phase6.")
    p.add_argument('--eval-on',choices=['test','val'],default='test')
    p.add_argument("--use-mentor-split", action=argparse.BooleanOptionalAction, default=True,
                   help="--no-use-mentor-split forces the legacy split; MUST match the split the "
                        "checkpoint was trained with, or the generated test set is wrong.")
    p.add_argument("--train-frac",    type=float, default=0.64)
    p.add_argument("--val-frac",      type=float, default=0.16)
    p.add_argument("--fold",          type=int, default=None,
                   help="CV fold index; generate for that fold's exact held-out test set.")
    p.add_argument("--n-folds",       type=int, default=5)
    p.add_argument("--seed",          type=int, default=None,
                   help="Seed torch/numpy for reproducible sampling (DDPM is stochastic; "
                        "without this every run gives slightly different metrics/figures).")
    p.add_argument("--guidance-scale", type=float, default=1.0,
                   help="Classifier-free guidance scale. 1.0 = plain conditional. >1 does a second "
                        "UNet pass with atrophy map + context zeroed and extrapolates. Needs a "
                        "checkpoint trained with --cfg-prob>0.")
    p.add_argument("--k",             type=int, default=1,
                   help="Samples per subject to average (posterior mean). k=1 = single sample; "
                        "k>1 averages k DDPM draws -> clean, low-noise map (no salt-and-pepper).")
    p.add_argument("--max-subjects",  type=int, default=None,
                   help="Only generate the first N test subjects (quick backfill demo; default: all).")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Seeded sampling: torch/np seed={args.seed}")

    device = torch.device(DEVICE)
    os.makedirs(args.generated_dir, exist_ok=True)

    ckpt      = torch.load(args.checkpoint, map_location=device)
    latent_ch = ckpt.get("latent_ch", args.latent_ch)
    ch_list   = tuple(int(x) for x in args.unet_channels.split(","))
    # Tissue + cond-mode: derive from the checkpoint so UNet/conditioner match.
    extra_cond_ch = ckpt.get("extra_cond_ch", 1)
    use_tissue    = ckpt.get("use_tissue", extra_cond_ch >= 4)
    use_spade     = ckpt.get("use_spade", False)
    cond_mode     = ckpt.get("cond_mode", "atrophy")
    cfg_prob      = ckpt.get("cfg_prob", 0.0)
    print(f"extra_cond_ch={extra_cond_ch}  use_tissue={use_tissue}  use_spade={use_spade}  "
          f"cond_mode={cond_mode}  cfg_prob={cfg_prob}")
    if args.guidance_scale != 1.0 and cfg_prob <= 0:
        print(f"WARNING: --guidance-scale={args.guidance_scale} but checkpoint cfg_prob={cfg_prob} "
              f"(NOT CFG-trained). Guided samples will be garbage — train with --cfg-prob>0.", flush=True)

    # Rebuild the AE at the architecture the checkpoint was trained with (legacy: 8x / 1 block).
    ae_scale = ckpt.get("ae_scale", 8)
    ae_rb    = ckpt.get("ae_res_blocks", 1)
    print(f"AE arch from checkpoint: scale={ae_scale}  res_blocks/level={ae_rb}")
    ae          = Autoencoder3D(latent_ch=latent_ch, scale=ae_scale, n_res_blocks=ae_rb).to(device)
    unet        = DenoisingUNet3D(latent_ch=latent_ch, ch_list=ch_list,
                                  n_transformer=args.n_transformer,
                                  extra_cond_ch=extra_cond_ch, use_spade=use_spade).to(device)
    if cond_mode == "none":
        conditioner = NullConditioner().to(device)
    elif cond_mode == "ptau217":
        conditioner = PTau217Conditioner(device=device)
    elif cond_mode == "ptau217_mlp":
        conditioner = PTau217MLPConditioner().to(device)
    else:
        conditioner = AtrophyConditioner().to(device)
    ae.load_state_dict(ckpt["ae"])
    # Prefer EMA weights when present (training saves "unet_ema" alongside raw "unet").
    unet_sd = ckpt.get("unet_ema") or ckpt["unet"]
    if "unet_ema" in ckpt:
        print("Using EMA UNet weights")
    unet.load_state_dict(unet_sd)
    if ckpt.get("conditioner"):
        conditioner.load_state_dict(ckpt["conditioner"])
    latent_std = ckpt.get("latent_std",
                          torch.ones(1, latent_ch, 1, 1, 1)).to(device)

    ae.eval(); unet.eval(); conditioner.eval()
    # Prefer an explicit --noise-schedule; otherwise use the schedule the checkpoint was
    # trained with (older checkpoints have no field → linear). A mismatch here silently
    # corrupts samples, so it must track training (e.g. exp_d trains with cosine).
    noise_sched = args.noise_schedule or ckpt.get("noise_schedule", "linear")
    schedule = DiffusionSchedule(device=device, schedule=noise_sched)
    print(f"Noise schedule: {noise_sched}"
          f"{' (from checkpoint)' if args.noise_schedule is None else ' (from --noise-schedule)'}")

    fold_kwargs = {} if args.fold is None else {"fold_idx": args.fold, "n_folds": args.n_folds}
    ds_cond_mode = "ptau217" if cond_mode == "ptau217_mlp" else cond_mode  # same data as ptau217
    if getattr(args, "phase6_manifest", None):
        from src import dataset_phase6 as _p6
        _p6cm = "combined" if cond_mode in ("combined", "combined_mlp") else ds_cond_mode
        _, _val_ds, test_ds, _, _, _ = _p6.build_dataloaders(
            manifest=args.phase6_manifest, use_dk_mask=args.use_mask,
            use_tissue=use_tissue, cond_mode=_p6cm, **fold_kwargs,
        )
    else:
        _, _val_ds, test_ds, _, _, _ = _dataset.build_dataloaders(
            use_dk_mask=args.use_mask,
            train_frac=args.train_frac, val_frac=args.val_frac,
            use_tissue=use_tissue, cond_mode=ds_cond_mode,
            use_mentor_split=args.use_mentor_split, **fold_kwargs,
        )
    if args.eval_on=='val': test_ds=_val_ds
    print(f"Test subjects: {len(test_ds)}")

    n_gen = len(test_ds) if args.max_subjects is None else min(len(test_ds), args.max_subjects)
    for i in tqdm(range(n_gen), desc="Generating spatial"):
        if use_tissue:
            pet, mri, atrophy_vec, atrophy_map, tissue, _ = test_ds[i]
        else:
            pet, mri, atrophy_vec, atrophy_map, _ = test_ds[i]
            tissue = None
        # k=1 -> single stochastic sample; k>1 -> posterior mean (average k DDPM draws),
        # which cancels per-voxel sampler noise (the salt-and-pepper) and is the model's
        # conditional-mean estimate.
        acc = None
        for _ in range(args.k):
            g = _ddpm_loop(ae, unet, conditioner, schedule, latent_std,
                           mri, atrophy_vec, atrophy_map, device,
                           n_steps=args.n_steps, tissue=tissue,
                           guidance_scale=args.guidance_scale)
            acc = g if acc is None else acc + g
        gen = (acc / args.k).astype(np.float32)
        np.save(os.path.join(args.generated_dir, f"subject_{i:03d}.npy"), gen)

    print(f"Saved {n_gen} files to {args.generated_dir}")


if __name__ == "__main__":
    main()
