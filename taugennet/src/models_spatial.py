#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
models_spatial.py – TauGenNet models for spatial atrophy conditioning (Option 2).

Identical to models.py except DenoisingUNet3D accepts LATENT_CH*2 + 1 = 7 input
channels: [z_t (3), z_m (3), atrophy_map_latent (1)].

The AE (Autoencoder3D) is unchanged and checkpoints from models.py are reusable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LATENT_CH, LATENT_SCALE, COND_DIM, DEVICE, GROUPNORM_GROUPS


# ── 3D Autoencoder (unchanged from models.py) ────────────────────────────────

class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(GROUPNORM_GROUPS, ch, eps=1e-6), nn.SiLU(),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(GROUPNORM_GROUPS, ch, eps=1e-6), nn.SiLU(),
        )
    def forward(self, x): return x + self.net(x)


class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, ch_mult=(64, 128, 128), latent_ch=LATENT_CH, scale=LATENT_SCALE,
                 n_res_blocks=1):
        super().__init__()
        n_down = int(math.log2(scale))
        assert len(ch_mult) == n_down
        layers = [nn.Conv3d(in_ch, ch_mult[0], 3, padding=1)]
        # n_res_blocks residual blocks per level (paper: 2). n_res_blocks=1 == original layout.
        for i in range(n_down):
            ch      = ch_mult[i]
            ch_next = ch_mult[i + 1] if i + 1 < n_down else ch
            layers += [ResBlock3D(ch) for _ in range(n_res_blocks)]
            layers += [nn.Conv3d(ch, ch_next, 4, stride=2, padding=1)]
        ch = ch_mult[-1]
        layers += [ResBlock3D(ch) for _ in range(n_res_blocks)]   # bottleneck level
        layers += [nn.GroupNorm(GROUPNORM_GROUPS, ch, eps=1e-6), nn.SiLU(),
                   nn.Conv3d(ch, latent_ch * 2, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        h = self.net(x)
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar


class Decoder3D(nn.Module):
    def __init__(self, latent_ch=LATENT_CH, ch_mult=(64, 128, 128), out_ch=1, scale=LATENT_SCALE,
                 n_res_blocks=1):
        super().__init__()
        n_up   = int(math.log2(scale))
        ch_rev = list(reversed(ch_mult))
        layers = [nn.Conv3d(latent_ch, ch_rev[0], 3, padding=1)]
        layers += [ResBlock3D(ch_rev[0]) for _ in range(n_res_blocks)]   # bottleneck level
        for i in range(n_up):
            ch      = ch_rev[i]
            ch_next = ch_rev[i + 1] if i + 1 < n_up else ch_rev[-1]
            layers += [nn.ConvTranspose3d(ch, ch_next, 4, stride=2, padding=1)]
            layers += [ResBlock3D(ch_next) for _ in range(n_res_blocks)]
        ch_final = ch_rev[-1]
        layers += [nn.GroupNorm(GROUPNORM_GROUPS, ch_final, eps=1e-6), nn.SiLU(),
                   nn.Conv3d(ch_final, out_ch, 3, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z): return self.net(z)


class Autoencoder3D(nn.Module):
    def __init__(self, latent_ch=LATENT_CH, scale=LATENT_SCALE, n_res_blocks=1, ch_mult=None):
        super().__init__()
        self.latent_ch    = latent_ch
        self.scale        = scale         # latent spatial downsampling (4/8/16)
        self.n_res_blocks = n_res_blocks  # residual blocks per level (paper: 2)
        n_down = int(math.log2(scale))
        if ch_mult is None:
            ch_mult = tuple([64] + [128] * (n_down - 1))   # scale=8 -> (64,128,128)
        self.ch_mult = ch_mult
        self.encoder = Encoder3D(latent_ch=latent_ch, ch_mult=ch_mult, scale=scale,
                                 n_res_blocks=n_res_blocks)
        self.decoder = Decoder3D(latent_ch=latent_ch, ch_mult=ch_mult, scale=scale,
                                 n_res_blocks=n_res_blocks)

    def encode(self, x):
        mean, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        z   = mean + std * torch.randn_like(std)
        return z, mean, logvar

    def encode_mean(self, x):
        mean, _ = self.encoder(x)
        return mean

    def decode(self, z): return self.decoder(z)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        return self.decode(z), mean, logvar


# ── Attention blocks (unchanged from models.py) ───────────────────────────────

class CrossAttention3D(nn.Module):
    def __init__(self, feat_dim, context_dim=COND_DIM, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = feat_dim // n_heads
        self.scale   = self.d_head ** -0.5
        self.norm    = nn.LayerNorm(feat_dim)
        self.to_q    = nn.Linear(feat_dim,    feat_dim, bias=False)
        self.to_k    = nn.Linear(context_dim, feat_dim, bias=False)
        self.to_v    = nn.Linear(context_dim, feat_dim, bias=False)
        self.to_out  = nn.Linear(feat_dim,    feat_dim)

    def forward(self, x, context):
        B, C, H, W, D = x.shape
        N = H * W * D
        x_flat = x.view(B, C, N).permute(0, 2, 1)
        x_norm = self.norm(x_flat)
        Q = self.to_q(x_norm)
        K = self.to_k(context)
        V = self.to_v(context)
        def split(t, s):
            return t.view(B, s, self.n_heads, self.d_head).transpose(1, 2)
        Q = split(Q, N); K = split(K, context.size(1)); V = split(V, context.size(1))
        A = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        O = torch.matmul(A, V).transpose(1, 2).contiguous().view(B, N, C)
        O = self.to_out(O)
        return (x_flat + O).permute(0, 2, 1).view(B, C, H, W, D)


class SelfAttention3D(nn.Module):
    def __init__(self, ch, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = ch // n_heads
        self.scale   = self.d_head ** -0.5
        self.norm    = nn.LayerNorm(ch)
        self.to_qkv  = nn.Linear(ch, ch * 3, bias=False)
        self.to_out  = nn.Linear(ch, ch)

    def forward(self, x):
        B, C, H, W, D = x.shape
        N = H * W * D
        x_flat = x.view(B, C, N).permute(0, 2, 1)
        x_norm = self.norm(x_flat)
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        Q, K, V = [t.view(B, N, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]
        A = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * self.scale, dim=-1)
        O = torch.matmul(A, V).transpose(1, 2).contiguous().view(B, N, C)
        return (x_flat + self.to_out(O)).permute(0, 2, 1).view(B, C, H, W, D)


class FeedForward3D(nn.Module):
    def __init__(self, ch, mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.net  = nn.Sequential(
            nn.Linear(ch, ch * mult), nn.SiLU(), nn.Linear(ch * mult, ch)
        )

    def forward(self, x):
        B, C, H, W, D = x.shape
        N = H * W * D
        x_flat = x.view(B, C, N).permute(0, 2, 1)
        return (x_flat + self.net(self.norm(x_flat))).permute(0, 2, 1).view(B, C, H, W, D)


class TransformerBlock3D(nn.Module):
    def __init__(self, ch, context_dim=COND_DIM, n_heads=8):
        super().__init__()
        self.sa  = SelfAttention3D(ch, n_heads)
        self.ca  = CrossAttention3D(ch, context_dim, n_heads)
        self.ffn = FeedForward3D(ch)

    def forward(self, x, ctx):
        x = self.sa(x)
        x = self.ca(x, ctx)
        x = self.ffn(x)
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb   = t[:, None].float() * freqs[None]
        emb   = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


class AtrophyEncoder3D(nn.Module):
    """Learned CNN encoder for the raw atrophy z-score map (spatial-injection Design B).

    (B, in_ch, 96,112,96) -> (B, out_ch, 12,14,12) via `n_down` stride-2 Conv3d+GN+SiLU
    blocks (n_down = log2(scale), so scale=8 -> 3 blocks), followed by a 1x1 projection
    conv. Replaces the unlearned `avg_pool3d(atrophy_map, kernel_size=scale)` baseline;
    out_ch=1 by default keeps the comparison apples-to-apples against that baseline.
    """
    def __init__(self, in_ch=1, width=32, out_ch=1, scale=LATENT_SCALE):
        super().__init__()
        n_down = int(math.log2(scale))
        layers = []
        c_in = in_ch
        for _ in range(n_down):
            layers += [nn.Conv3d(c_in, width, 4, stride=2, padding=1),
                       nn.GroupNorm(min(GROUPNORM_GROUPS, width), width, eps=1e-6),
                       nn.SiLU()]
            c_in = width
        layers += [nn.Conv3d(width, out_ch, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SPADE3D(nn.Module):
    """Spatially-adaptive normalization (spatial-injection Design A): param-free
    GroupNorm, then per-voxel gamma/beta predicted from the atrophy map resampled
    (nearest) to this block's resolution. h_out = norm(h) * (1 + gamma) + beta.

    gamma/beta projections are zero-initialized so an untrained SPADE layer starts
    as an exact no-op (norm(h) passes through unchanged) — without this, an
    untrained layer injects noise into an otherwise-stable UNet from epoch 0 and
    looks like a regression unrelated to the actual idea being tested.
    """
    def __init__(self, ch, map_ch=1, hidden=64):
        super().__init__()
        self.norm      = nn.GroupNorm(min(GROUPNORM_GROUPS, ch), ch, eps=1e-6, affine=False)
        self.shared    = nn.Sequential(nn.Conv3d(map_ch, hidden, 3, padding=1), nn.SiLU())
        self.to_gamma  = nn.Conv3d(hidden, ch, 3, padding=1)
        self.to_beta   = nn.Conv3d(hidden, ch, 3, padding=1)
        nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

    def forward(self, h, atrophy_map):
        m = F.interpolate(atrophy_map, size=h.shape[2:], mode='nearest')
        s = self.shared(m)
        gamma, beta = self.to_gamma(s), self.to_beta(s)
        return self.norm(h) * (1 + gamma) + beta


class UNetBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, context_dim=COND_DIM, n_transformer=3,
                 downsample_ch=None, upsample_ch=None, upsample_in_ch=None,
                 use_spade=False, spade_hidden=64):
        super().__init__()
        self.use_spade = use_spade
        self.norm1   = SPADE3D(out_ch, map_ch=1, hidden=spade_hidden) if use_spade \
                       else nn.GroupNorm(GROUPNORM_GROUPS, out_ch, eps=1e-6)
        self.conv1   = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.t_proj1 = nn.Linear(t_dim, out_ch)
        self.norm2   = nn.GroupNorm(GROUPNORM_GROUPS, out_ch, eps=1e-6)
        self.conv2   = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip1   = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.trans1  = nn.ModuleList([
            TransformerBlock3D(out_ch, context_dim) for _ in range(n_transformer)
        ])
        self.norm3   = nn.GroupNorm(GROUPNORM_GROUPS, out_ch, eps=1e-6)
        self.conv3   = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.t_proj2 = nn.Linear(t_dim, out_ch)
        self.norm4   = nn.GroupNorm(GROUPNORM_GROUPS, out_ch, eps=1e-6)
        self.conv4   = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.trans2  = nn.ModuleList([
            TransformerBlock3D(out_ch, context_dim) for _ in range(n_transformer)
        ])
        self.downsample = nn.Conv3d(out_ch, downsample_ch, 4, stride=2, padding=1) \
                          if downsample_ch is not None else None
        _up_in = upsample_in_ch if upsample_in_ch is not None else out_ch
        self.upsample        = nn.ConvTranspose3d(_up_in, upsample_ch, 4, stride=2, padding=1) \
                               if upsample_ch is not None else None
        self._upsample_in_ch = upsample_in_ch

    def forward(self, x, t_emb, ctx, atrophy_map=None):
        c1 = self.conv1(x)
        h  = F.silu(self.norm1(c1, atrophy_map) if self.use_spade else self.norm1(c1))
        h  = h + self.t_proj1(F.silu(t_emb))[:, :, None, None, None]
        h  = F.silu(self.norm2(self.conv2(h))) + self.skip1(x)
        for blk in self.trans1:
            h = blk(h, ctx)
        r  = F.silu(self.norm3(self.conv3(h)))
        r  = r + self.t_proj2(F.silu(t_emb))[:, :, None, None, None]
        h  = h + F.silu(self.norm4(self.conv4(r)))
        for blk in self.trans2:
            h = blk(h, ctx)
        skip = h
        if self.downsample is not None:
            return skip, self.downsample(h)
        if self.upsample is not None:
            if self._upsample_in_ch is not None:
                return skip, self.upsample(x[:, :self._upsample_in_ch])
            return skip, self.upsample(h)
        return skip, h


# ── Spatial Denoising U-Net ───────────────────────────────────────────────────

class DenoisingUNet3D(nn.Module):
    """Denoising U-Net for spatial atrophy (+ optional tissue) conditioning.

    Input: h_t = cat(z_t, z_m, atrophy_latent[, tissue_latent])
           shape (B, LATENT_CH*2 + extra_cond_ch, lH, lW, lD)
    Output: predicted noise                       shape (B, LATENT_CH, lH, lW, lD)

    extra_cond_ch counts the spatial conditioning channels appended after
    [z_t, z_m]: 1 = atrophy map only (default); 4 = atrophy(1) + tissue one-hot(3).
    Each is the corresponding full-res map downsampled to latent resolution
    (12×14×12) via avg_pool3d(kernel_size=8) — unless `use_atrophy_encoder=True`,
    in which case the atrophy channel is produced by a learned `AtrophyEncoder3D`
    instead (spatial-injection Design B); any tissue channels still use avg_pool3d.

    `use_spade=True` (spatial-injection Design A) additionally feeds the RAW
    atrophy_map into a SPADE layer at every enc/mid/dec block (resampled to each
    block's own resolution), independent of `use_atrophy_encoder` — the two are
    orthogonal flags and can be combined.
    """
    def __init__(self, latent_ch=LATENT_CH, ch_list=(256, 512, 768),
                 t_dim=256, context_dim=COND_DIM, n_transformer=3,
                 extra_cond_ch=1, use_atrophy_encoder=False, atrophy_encoder_width=32,
                 use_spade=False, spade_hidden=64):
        in_ch = latent_ch * 2 + extra_cond_ch
        super().__init__()
        c1, c2, c3 = ch_list
        nt = n_transformer

        self.use_atrophy_encoder = use_atrophy_encoder
        self.use_spade  = use_spade
        self._tissue_ch = extra_cond_ch - 1 if extra_cond_ch >= 4 else 0
        self.atrophy_encoder = (
            AtrophyEncoder3D(in_ch=1, width=atrophy_encoder_width, out_ch=1)
            if use_atrophy_encoder else None
        )

        self.t_embed = TimestepEmbedding(t_dim)
        self.in_conv = nn.Conv3d(in_ch, c1, 3, padding=1)   # latent_ch*2 + extra_cond_ch

        sp = dict(use_spade=use_spade, spade_hidden=spade_hidden)
        self.enc1  = UNetBlock3D(c1, c1, t_dim, context_dim, nt, downsample_ch=c2, **sp)
        self.enc2  = UNetBlock3D(c2, c2, t_dim, context_dim, nt, downsample_ch=c3, **sp)
        self.enc3  = UNetBlock3D(c3, c3, t_dim, context_dim, nt, downsample_ch=c3, **sp)
        self.mid   = UNetBlock3D(c3, c3, t_dim, context_dim, nt, **sp)

        self.up3 = nn.ConvTranspose3d(c3, c3, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1)

        self.dec3  = UNetBlock3D(c3 * 2, c3, t_dim, context_dim, nt, **sp)
        self.dec2  = UNetBlock3D(c2 * 2, c2, t_dim, context_dim, nt, **sp)
        self.dec1  = UNetBlock3D(c1 * 2, c1, t_dim, context_dim, nt, **sp)

        self.out   = nn.Sequential(
            nn.GroupNorm(GROUPNORM_GROUPS, c1, eps=1e-6), nn.SiLU(), nn.Conv3d(c1, latent_ch, 1)
        )

    def forward(self, zt, zm, t, ctx, atrophy_map, tissue_map=None):
        if self.use_atrophy_encoder:
            atrophy_latent = self.atrophy_encoder(atrophy_map)             # (B,1,lH,lW,lD), grad flows
        else:
            atrophy_latent = F.avg_pool3d(atrophy_map, kernel_size=LATENT_SCALE)
        if self._tissue_ch > 0:
            tissue_latent = F.avg_pool3d(tissue_map, kernel_size=LATENT_SCALE)
            cond_latent   = torch.cat([atrophy_latent, tissue_latent], dim=1)
        else:
            cond_latent   = atrophy_latent
        ht = torch.cat([zt, zm, cond_latent], dim=1)

        t_emb = self.t_embed(t)
        x  = self.in_conv(ht)

        # SPADE's own atrophy_map input is always the RAW map (not the atrophy_encoder's
        # output), by design — it isolates "one vs many injection points" as an independent
        # variable from Design B's "learned vs raw" (see SPATIAL_INJECTION_CONTEXT.md).
        amap = atrophy_map if self.use_spade else None

        e1, x2 = self.enc1(x,  t_emb, ctx, amap)
        e2, x3 = self.enc2(x2, t_emb, ctx, amap)
        e3, xm = self.enc3(x3, t_emb, ctx, amap)
        _,  m  = self.mid(xm, t_emb, ctx, amap)

        u3 = F.interpolate(self.up3(m),  size=e3.shape[2:], mode='trilinear', align_corners=False)
        u2 = F.interpolate(self.up2(u3), size=e2.shape[2:], mode='trilinear', align_corners=False)
        u1 = F.interpolate(self.up1(u2), size=e1.shape[2:], mode='trilinear', align_corners=False)

        _, d3 = self.dec3(torch.cat([u3, e3], dim=1), t_emb, ctx, amap)
        _, d2 = self.dec2(torch.cat([u2, e2], dim=1), t_emb, ctx, amap)
        _, d1 = self.dec1(torch.cat([u1, e1], dim=1), t_emb, ctx, amap)

        return self.out(d1)
