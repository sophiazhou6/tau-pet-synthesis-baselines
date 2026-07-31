#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
models_dynamic_prompt.py – TauGenNet with CoMA-UNet dynamic prompt head (Option 1).

Adds a learnable spatial prompt module on top of the standard DenoisingUNet3D.
The UNet input is unchanged (in_ch = LATENT_CH*2 = 6); the prompt operates as a
post-UNet modulation in latent space.

Architecture adapted from CoMA-UNet (Borhi et al.)
  - StackedFusionConvLayers: MONAI-free port of lines 480–501
  - DenoisingUNet3DWithPrompt: subclasses DenoisingUNet3D from models.py,
    adds pos/neg/general prompts + deep_mod + fusion + pred_head

Key differences vs. CoMA-UNet:
  - CatBoost saliency maps replaced by the atlas-based atrophy_map (from dataset_spatial.py)
  - pos/neg selection based on AD/MCI diagnosis (1=AD, 0=MCI) instead of amyloid-PET
  - Prompt size matches latent resolution (LATENT_CH, 12, 14, 12) not 128³
  - No MONAI dependency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LATENT_CH, COND_DIM, DEVICE
from .models import DenoisingUNet3D   # base UNet unchanged; in_ch=6


# ── StackedFusionConvLayers (ported from CoMA-UNet, MONAI-free) ──────────────

class StackedFusionConvLayers(nn.Module):
    """Stacked conv bottleneck: in_ch → bottleneck × (n_convs-2) → out_ch.

    Ported from attn_unet_data_parallel.py lines 480–501 with MONAI replaced by
    standard nn.Conv3d + nn.LeakyReLU.
    """

    def __init__(self, in_ch, out_ch, bottleneck, n_convs=3,
                 negative_slope=1e-2):
        super().__init__()
        act = lambda: nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        layers = [nn.Conv3d(in_ch, bottleneck, 3, padding=1), act()]
        for _ in range(n_convs - 2):
            layers += [nn.Conv3d(bottleneck, bottleneck, 3, padding=1), act()]
        layers += [nn.Conv3d(bottleneck, out_ch, 3, padding=1), act()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Dynamic prompt UNet ───────────────────────────────────────────────────────

class DenoisingUNet3DWithPrompt(DenoisingUNet3D):
    """DenoisingUNet3D with a CoMA-style dynamic prompt modulation head.

    The base UNet runs first (in_ch=6, unchanged). Then:
      1. A pos or neg learnable spatial prompt is selected per sample (AD/MCI).
      2. The selected prompt is modulated by the atlas-based atrophy map + UNet output
         via deep_mod (3-input stacked conv → delta).
      3. modulated = general_prompt + deep_mod(prompt, atrophy_map, out)
      4. final_out = pred_head( cat([out, fusion(cat([modulated, out]))]) )

    All prompt parameters are at latent resolution (LATENT_CH, lH, lW, lD).

    Forward signature:
        unet(ht, t, ctx, atrophy_map, diagnosis)
          ht          : (B, 6, lH, lW, lD)
          t           : (B,) long
          ctx         : (B, n_tokens, COND_DIM)
          atrophy_map : (B, 1, lH, lW, lD)  — already at latent resolution
          diagnosis   : (B,) long  — 1=AD (pos_prompt), 0=MCI (neg_prompt)
    """

    def __init__(self, latent_ch=LATENT_CH, ch_list=(256, 512, 768), t_dim=256,
                 context_dim=COND_DIM, n_transformer=3):
        super().__init__(
            latent_ch=latent_ch,
            ch_list=ch_list,
            t_dim=t_dim,
            context_dim=context_dim,
            n_transformer=n_transformer,
        )
        self.latent_ch = latent_ch
        scale = 0.01
        self.pos_prompt     = nn.Parameter(scale * torch.randn(1, latent_ch, 12, 14, 12))
        self.neg_prompt     = nn.Parameter(scale * torch.randn(1, latent_ch, 12, 14, 12))
        self.general_prompt = nn.Parameter(scale * torch.randn(1, latent_ch, 12, 14, 12))

        self.deep_mod = StackedFusionConvLayers(
            in_ch=latent_ch * 2 + 1, out_ch=latent_ch, bottleneck=16, n_convs=3
        )
        self.fusion = StackedFusionConvLayers(
            in_ch=latent_ch * 2, out_ch=latent_ch, bottleneck=8, n_convs=3
        )
        self.pred_head = nn.Conv3d(latent_ch * 2, latent_ch, kernel_size=1)

    def forward(self, ht, t, ctx, atrophy_map=None, diagnosis=None):
        out = super().forward(ht, t, ctx)   # (B, LATENT_CH, lH, lW, lD)

        if atrophy_map is not None and diagnosis is not None:
            out = self._apply_prompt(out, atrophy_map, diagnosis)

        return out

    def _apply_prompt(self, out, atrophy_map, diagnosis):
        """
        out         : (B, LATENT_CH, lH, lW, lD)
        atrophy_map : (B, 1, lH, lW, lD)
        diagnosis   : (B,) long — 1=AD, 0=MCI
        """
        B = out.size(0)

        # Select pos/neg prompt per sample, stack into (B, LATENT_CH, lH, lW, lD)
        prompt_list = []
        for b in range(B):
            p = self.pos_prompt if diagnosis[b].item() == 1 else self.neg_prompt
            prompt_list.append(p.expand(1, -1, -1, -1, -1))
        prompt = torch.cat(prompt_list, dim=0)   # (B, LATENT_CH, lH, lW, lD)

        # Resize atrophy_map to match out's spatial dims (handles minor size mismatches)
        if atrophy_map.shape[2:] != out.shape[2:]:
            atrophy_map = F.interpolate(atrophy_map.float(), size=out.shape[2:],
                                        mode="trilinear", align_corners=False)

        # Modulate: general_prompt + deep_mod(prompt, atrophy_map, out)
        mod_input  = torch.cat([prompt, atrophy_map, out], dim=1)   # (B, 2L+1, ...)
        modulated  = self.general_prompt.expand(B, -1, -1, -1, -1) + self.deep_mod(mod_input)

        # Fuse: fusion(modulated, out) → residual correction
        fused = self.fusion(torch.cat([modulated, out], dim=1))      # (B, LATENT_CH, ...)

        # Final prediction
        return self.pred_head(torch.cat([out, fused], dim=1))        # (B, LATENT_CH, ...)
