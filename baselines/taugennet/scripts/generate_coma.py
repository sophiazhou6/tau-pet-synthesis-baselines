#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""Generate tau PET for CoMA dynamic-prompt model.

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
from src.models                import Autoencoder3D
from src.models_dynamic_prompt import DenoisingUNet3DWithPrompt
from src.diffusion             import DiffusionSchedule
from src.conditioning          import AtrophyConditioner


@torch.no_grad()
def _ddpm_loop(ae, unet, conditioner, schedule, latent_std,
               mri, atrophy_vec, atrophy_map, diagnosis, device, n_steps=500):
    """Single-sample DDPM for CoMA model.

    ht = cat(zt, zm) — standard 2*latent_ch channels; atrophy_map and
    diagnosis are passed as extra kwargs to the UNet forward (prompt modulation).
    """
    mri_b  = mri.unsqueeze(0).to(device)
    avec   = atrophy_vec.unsqueeze(0).to(device)
    amap   = atrophy_map.unsqueeze(0).to(device)  # (1,1,H,W,D)
    diag_b = torch.tensor([diagnosis], device=device, dtype=torch.long)

    zm  = ae.encode_mean(mri_b) / latent_std
    ctx = conditioner.encode(avec)
    zt  = torch.randn_like(zm)

    am_latent = F.avg_pool3d(amap, kernel_size=8)  # (1,1,12,14,12)

    step = T_STEPS // n_steps
    ts   = list(reversed(range(0, T_STEPS, step)))
    for i, t_idx in enumerate(ts):
        t_batch = torch.full((1,), t_idx, device=device, dtype=torch.long)
        ht      = torch.cat([zt, zm], dim=1)
        eps_hat = unet(ht, t_batch, ctx, atrophy_map=am_latent, diagnosis=diag_b)

        sqrt_ab       = schedule.sqrt_ab[t_idx]
        sqrt_one_m_ab = schedule.sqrt_one_m_ab[t_idx]
        z0_hat = (zt - sqrt_one_m_ab * eps_hat) / sqrt_ab

        if t_idx == 0:
            zt = z0_hat
        else:
            beta_t  = schedule.betas[t_idx]
            alpha_t = schedule.alphas[t_idx]
            ab_prev = schedule.alpha_bar_prev[t_idx]
            coef1   = ab_prev.sqrt() * beta_t / (1 - schedule.alpha_bar[t_idx])
            coef2   = alpha_t.sqrt() * (1 - ab_prev) / (1 - schedule.alpha_bar[t_idx])
            mean    = coef1 * z0_hat + coef2 * zt
            zt      = mean + schedule.post_var[t_idx].sqrt() * torch.randn_like(zt)

    return ae.decode(zt * latent_std).squeeze().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True,
                   help="Path to CoMA diffusion checkpoint .pt")
    p.add_argument("--generated-dir", required=True,
                   help="Directory to save subject_NNN.npy files")
    p.add_argument("--latent-ch",     type=int, default=LATENT_CH)
    p.add_argument("--unet-channels", type=str, default="128,256,512")
    p.add_argument("--n-transformer", type=int, default=1)
    p.add_argument("--n-steps",       type=int, default=500)
    p.add_argument("--use-mask",      action="store_true", default=True)
    p.add_argument("--train-frac",    type=float, default=0.64)
    p.add_argument("--val-frac",      type=float, default=0.16)
    args = p.parse_args()

    device = torch.device(DEVICE)
    os.makedirs(args.generated_dir, exist_ok=True)

    ckpt      = torch.load(args.checkpoint, map_location=device)
    latent_ch = ckpt.get("latent_ch", args.latent_ch)
    ch_list   = tuple(int(x) for x in args.unet_channels.split(","))

    ae          = Autoencoder3D(latent_ch=latent_ch).to(device)
    unet        = DenoisingUNet3DWithPrompt(latent_ch=latent_ch, ch_list=ch_list,
                                            n_transformer=args.n_transformer).to(device)
    conditioner = AtrophyConditioner().to(device)
    ae.load_state_dict(ckpt["ae"])
    unet.load_state_dict(ckpt["unet"])
    if ckpt.get("conditioner"):
        conditioner.load_state_dict(ckpt["conditioner"])
    latent_std = ckpt.get("latent_std",
                          torch.ones(1, latent_ch, 1, 1, 1)).to(device)

    ae.eval(); unet.eval(); conditioner.eval()
    schedule = DiffusionSchedule(device=device)

    _, _, test_ds, _, _, _ = _dataset.build_dataloaders(
        use_dk_mask=args.use_mask,
        train_frac=args.train_frac, val_frac=args.val_frac,
    )
    print(f"Test subjects: {len(test_ds)}")

    for i in tqdm(range(len(test_ds)), desc="Generating CoMA"):
        pet, mri, atrophy_vec, atrophy_map, diagnosis = test_ds[i]
        gen = _ddpm_loop(ae, unet, conditioner, schedule, latent_std,
                         mri, atrophy_vec, atrophy_map, int(diagnosis),
                         device, n_steps=args.n_steps)
        np.save(os.path.join(args.generated_dir, f"subject_{i:03d}.npy"), gen)

    print(f"Saved {len(test_ds)} files to {args.generated_dir}")


if __name__ == "__main__":
    main()
