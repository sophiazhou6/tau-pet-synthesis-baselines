import torch
from tqdm import tqdm

from .config import DEVICE, T_STEPS, DIFF_CHECKPOINT_PATH, AE_CHECKPOINT_PATH, LATENT_CH
from .diffusion import DiffusionSchedule
from .models import Autoencoder3D, DenoisingUNet3D
from .conditioning import build_conditioner


def _remap_unet_keys(state_dict):
    """Remap standalone down modules to enc block submodule naming.

    Checkpoint uses standalone down1/down2/down3 keys; model stores them as
    enc1.downsample/enc2.downsample/enc3.downsample.
    up1/up2/up3 stay as-is — they map directly to self.up1/up2/up3 standalone attrs.
    """
    remap = {
        "down1": "enc1.downsample",
        "down2": "enc2.downsample",
        "down3": "enc3.downsample",
    }
    new_sd = {}
    for k, v in state_dict.items():
        prefix = k.split(".")[0]
        if prefix in remap:
            new_sd[k.replace(prefix, remap[prefix], 1)] = v
        else:
            new_sd[k] = v
    return new_sd


def load_models(checkpoint_path, mode, device=DEVICE, arch='silu',
                ch_list=(256, 512, 768), n_transformer=3):
    """Load ae, unet, conditioner, and latent_std from a checkpoint.

    checkpoint_path : path to the diffusion checkpoint
    mode            : 'atrophy' or 'ptau217'
    arch            : 'silu' (default) or 'relu' — must match the checkpoint's training variant
    ch_list         : UNet channel widths — must match the checkpoint's training config
    n_transformer   : transformer blocks per UNet level — must match training config
    Returns         : (ae, unet, conditioner, latent_std, diff_losses)

    latent_std defaults to ones if absent (old checkpoints without scaling).
    """
    if arch == 'relu':
        from .models_relu import Autoencoder3D as _AE, DenoisingUNet3D as _UNet
    elif arch == 'spatial':
        # NOTE: stale/incomplete for the spatial-conditioning family — extra_cond_ch is
        # never passed here (silently defaults to 1) and synthesize_tau_pet() has no
        # atrophy_map param, so this path cannot actually run DenoisingUNet3D.forward()
        # for arch='spatial' (which now requires atrophy_map). The real eval path is
        # generate_spatial.py + evaluate_final.py --use-cached; this branch is unused.
        from .models_spatial import Autoencoder3D as _AE, DenoisingUNet3D as _UNet
    elif arch == 'coma':
        from .models import Autoencoder3D as _AE
        from .models_dynamic_prompt import DenoisingUNet3DWithPrompt as _UNet
    else:
        from .models import Autoencoder3D as _AE, DenoisingUNet3D as _UNet
    ckpt        = torch.load(checkpoint_path, map_location=device)
    latent_ch   = ckpt.get("latent_ch", LATENT_CH)
    _ae_kw = {"latent_ch": latent_ch}
    if arch not in ("relu", "spatial"):
        if ckpt.get("ae_scale") is not None:
            _ae_kw["scale"] = ckpt["ae_scale"]
        if ckpt.get("ae_res_blocks") is not None:
            _ae_kw["n_res_blocks"] = ckpt["ae_res_blocks"]
    ae          = _AE(**_ae_kw).to(device)
    unet        = _UNet(latent_ch=latent_ch, ch_list=ch_list, n_transformer=n_transformer).to(device)
    conditioner = build_conditioner(mode, device=device)

    ae.load_state_dict(ckpt["ae"])
    # Prefer EMA weights when present (training saves "unet_ema" alongside raw "unet").
    unet_sd = ckpt.get("unet_ema") or ckpt["unet"]
    if "unet_ema" in ckpt:
        print("Using EMA UNet weights")
    unet.load_state_dict(_remap_unet_keys(unet_sd))
    if ckpt.get("conditioner"):
        conditioner.load_state_dict(ckpt["conditioner"])

    # latent_std: (1, latent_ch, 1, 1, 1) — ones for backwards-compat with old ckpts
    latent_std  = ckpt.get("latent_std", torch.ones(1, latent_ch, 1, 1, 1)).to(device)
    diff_losses = ckpt.get("diff_losses", [])
    print(f"Loaded {mode} model: {len(diff_losses)} diffusion epochs")
    return ae, unet, conditioner, latent_std, diff_losses


def _prepare_cond(cond_data, device):
    """Ensure cond_data is a (1, N) tensor on device."""
    if not isinstance(cond_data, torch.Tensor):
        cond_data = torch.tensor(cond_data, dtype=torch.float32)
    if cond_data.dim() == 1:
        cond_data = cond_data.unsqueeze(0)  # (N,) → (1, N)
    return cond_data.to(device)


@torch.no_grad()
def synthesize_tau_pet(mri_vol, cond_data, ae, unet, schedule, encode_cond,
                       latent_std, device=DEVICE, n_steps=500, sampler='ddpm',
                       batch_size=8):
    """
    mri_vol    : (B, 1, H, W, D) or (1, 1, H, W, D) normalised MRI tensor
    cond_data  : (B, 86) atrophy z-scores  [atrophy mode]
                 (B, 1)  p-tau217 values   [ptau217 mode]
                 1-D inputs are unsqueezed to batch dim 1.
    latent_std : (1, LATENT_CH, 1, 1, 1) per-channel std used to normalise latents
    encode_cond: conditioner.encode callable
    batch_size : max samples per forward pass (default 8, matches training)
    sampler    : 'ddim' (fast, good at 50 steps) or 'ddpm' (best at 500+ steps)
    Returns    : (B, 1, H, W, D) synthesised tau PET on CPU
    """
    ae.eval(); unet.eval()

    mri_vol   = mri_vol.to(device)
    cond_data = _prepare_cond(cond_data, device)
    B         = mri_vol.shape[0]
    step      = T_STEPS // n_steps
    ts        = list(reversed(range(0, T_STEPS, step)))

    out_chunks = []
    for start in range(0, B, batch_size):
        mri_b  = mri_vol[start:start + batch_size]
        cond_b = cond_data[start:start + batch_size]
        zm     = ae.encode_mean(mri_b) / latent_std
        c      = encode_cond(cond_b)
        zt     = torch.randn_like(zm)
        for i, t_idx in enumerate(tqdm(ts, desc=f"Denoising [{start}:{start+len(mri_b)}]", leave=False)):
            if sampler == 'ddim':
                t_prev = ts[i + 1] if i + 1 < len(ts) else -1
                zt = schedule.ddim_sample(unet, zt, t_idx, t_prev, zm, c)
            else:
                zt = schedule.p_sample(unet, zt, t_idx, zm, c)
        out_chunks.append(ae.decode(zt * latent_std).cpu())

    return torch.cat(out_chunks, dim=0)


@torch.no_grad()
def synthesize_no_mri(mri_vol, cond_data, ae, unet, schedule, encode_cond,
                      latent_std, device=DEVICE, n_steps=500):
    """Ablation: MRI latent zeroed out — conditioning only, no structural guidance."""
    ae.eval(); unet.eval()
    zm_zero = torch.zeros_like(ae.encode_mean(mri_vol.to(device)))
    c       = encode_cond(_prepare_cond(cond_data, device))
    zt      = torch.randn_like(zm_zero)
    step    = T_STEPS // n_steps
    ts      = list(reversed(range(0, T_STEPS, step)))
    for i, t_idx in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        zt = schedule.ddim_sample(unet, zt, t_idx, t_prev, zm_zero, c)
    return ae.decode(zt * latent_std).cpu()
