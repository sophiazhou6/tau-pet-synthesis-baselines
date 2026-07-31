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
from tqdm import tqdm

from src.config import DEVICE, T_STEPS, LATENT_CH
from src import dataset_spatial as _dataset
from src.models_spatial   import Autoencoder3D, DenoisingUNet3D
from src.diffusion         import DiffusionSchedule
from src.conditioning      import AtrophyConditioner, NullConditioner, PTau217Conditioner


@torch.no_grad()
def _ddpm_loop(ae, unet, conditioner, schedule, latent_std,
               mri, atrophy_vec, atrophy_map, device, n_steps=500, tissue=None):
    """Single-sample DDPM for spatial model: ht = cat(zt, zm, cond_latent),
    where cond_latent is the atrophy map (+ tissue one-hot if provided)."""
    mri_b = mri.unsqueeze(0).to(device)
    avec  = atrophy_vec.unsqueeze(0).to(device)
    amap  = atrophy_map.unsqueeze(0).to(device)  # (1,1,H,W,D)

    zm  = ae.encode_mean(mri_b) / latent_std
    ctx = conditioner.encode(avec)
    zt  = torch.randn_like(zm)

    tissue_b = tissue.unsqueeze(0).to(device) if tissue is not None else None

    step = T_STEPS // n_steps
    ts   = list(reversed(range(0, T_STEPS, step)))
    for i, t_idx in enumerate(ts):
        t_batch = torch.full((1,), t_idx, device=device, dtype=torch.long)
        eps_hat = unet(zt, zm, t_batch, ctx, amap, tissue_map=tissue_b)

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
    ae_res_blocks = ckpt.get("ae_res_blocks", 1)
    ch_list   = tuple(int(x) for x in args.unet_channels.split(","))
    # Tissue + cond-mode: derive from the checkpoint so UNet/conditioner match.
    extra_cond_ch = ckpt.get("extra_cond_ch", 1)
    use_tissue    = ckpt.get("use_tissue", extra_cond_ch >= 4)
    cond_mode     = ckpt.get("cond_mode", "atrophy")
    use_atrophy_encoder   = ckpt.get("use_atrophy_encoder", False)
    atrophy_encoder_width = ckpt.get("atrophy_encoder_width", 32) or 32
    use_spade             = ckpt.get("use_spade", False)
    spade_hidden          = ckpt.get("spade_hidden", 64) or 64
    print(f"extra_cond_ch={extra_cond_ch}  use_tissue={use_tissue}  cond_mode={cond_mode}"
          f"  use_atrophy_encoder={use_atrophy_encoder}  use_spade={use_spade}"
          f"  ae_res_blocks={ae_res_blocks}")

    ae          = Autoencoder3D(latent_ch=latent_ch, n_res_blocks=ae_res_blocks).to(device)
    unet        = DenoisingUNet3D(latent_ch=latent_ch, ch_list=ch_list,
                                  n_transformer=args.n_transformer,
                                  extra_cond_ch=extra_cond_ch,
                                  use_atrophy_encoder=use_atrophy_encoder,
                                  atrophy_encoder_width=atrophy_encoder_width,
                                  use_spade=use_spade,
                                  spade_hidden=spade_hidden).to(device)
    if cond_mode == "none":
        conditioner = NullConditioner().to(device)
    elif cond_mode == "ptau217":
        conditioner = PTau217Conditioner(device=device)
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
    _, _, test_ds, _, _, _ = _dataset.build_dataloaders(
        use_dk_mask=args.use_mask,
        train_frac=args.train_frac, val_frac=args.val_frac,
        use_tissue=use_tissue, cond_mode=cond_mode,
        use_mentor_split=args.use_mentor_split, **fold_kwargs,
    )
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
                           n_steps=args.n_steps, tissue=tissue)
            acc = g if acc is None else acc + g
        gen = (acc / args.k).astype(np.float32)
        np.save(os.path.join(args.generated_dir, f"subject_{i:03d}.npy"), gen)

    print(f"Saved {n_gen} files to {args.generated_dir}")


if __name__ == "__main__":
    main()
