"""
models_resblock_diff.py — variant of models.py with post-activation ResBlock3D ordering.

Change from models.py:
  ResBlock3D ordering: GN → SiLU → Conv (pre-activation)
                    →  Conv → GN  → SiLU (post-activation, standard ResNet style)

Everything else (Encoder3D, Decoder3D, Autoencoder3D, DenoisingUNet3D) is identical.
Import this module instead of models.py to test the ordering difference.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from .config import LATENT_CH, LATENT_SCALE, COND_DIM, DEVICE, CLIP_MODEL_PATH, GROUPNORM_GROUPS


# ── 3D Autoencoder ────────────────────────────────────────────────────────────

class ResBlock3D(nn.Module):
    # Post-activation ordering: Conv → GN → SiLU (standard ResNet style).
    # models.py uses pre-activation: GN → SiLU → Conv (as in pre-act ResNet / DDPM).
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
    def __init__(self, in_ch=1, ch_mult=(64, 128, 128), latent_ch=LATENT_CH, scale=LATENT_SCALE):
        super().__init__()
        n_down = int(math.log2(scale))
        assert len(ch_mult) == n_down, f"ch_mult must have {n_down} entries for scale={scale}"

        layers = [nn.Conv3d(in_ch, ch_mult[0], 3, padding=1)]

        for i in range(n_down):
            ch      = ch_mult[i]
            ch_next = ch_mult[i + 1] if i + 1 < n_down else ch
            layers += [ResBlock3D(ch), nn.Conv3d(ch, ch_next, 4, stride=2, padding=1)]

        ch = ch_mult[-1]
        layers += [ResBlock3D(ch), nn.GroupNorm(GROUPNORM_GROUPS, ch, eps=1e-6), nn.SiLU(),
                   nn.Conv3d(ch, latent_ch * 2, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        h = self.net(x)
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar


class Decoder3D(nn.Module):
    def __init__(self, latent_ch=LATENT_CH, ch_mult=(64, 128, 128), out_ch=1, scale=LATENT_SCALE):
        super().__init__()
        n_up   = int(math.log2(scale))
        ch_rev = list(reversed(ch_mult))

        layers = [nn.Conv3d(latent_ch, ch_rev[0], 3, padding=1), ResBlock3D(ch_rev[0])]

        for i in range(n_up):
            ch      = ch_rev[i]
            ch_next = ch_rev[i + 1] if i + 1 < n_up else ch_rev[-1]
            layers += [nn.ConvTranspose3d(ch, ch_next, 4, stride=2, padding=1), ResBlock3D(ch_next)]

        ch_final = ch_rev[-1]
        layers += [nn.GroupNorm(GROUPNORM_GROUPS, ch_final, eps=1e-6), nn.SiLU(),
                   nn.Conv3d(ch_final, out_ch, 3, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z): return self.net(z)


class Autoencoder3D(nn.Module):
    """VAE-style autoencoder shared by PET and MRI (paper Eq. 1-3).

    encode()      – reparameterised sample; used during AE pretraining.
    encode_mean() – deterministic mean; used for z0/zm during diffusion training
                    and inference so conditioning is stable across calls.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder3D()
        self.decoder = Decoder3D()

    def encode(self, x):
        mean, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        z   = mean + std * torch.randn_like(std)
        return z, mean, logvar

    def encode_mean(self, x):
        """Deterministic encoding — returns the distribution mean, no sampling."""
        mean, _ = self.encoder(x)
        return mean

    def decode(self, z): return self.decoder(z)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        return self.decode(z), mean, logvar


# ── Denoising U-Net with Cross-Attention ─────────────────────────────────────

class CrossAttention3D(nn.Module):
    """Spatial-token × context cross-attention (Eq. 8–12)."""
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
    """Multi-head self-attention over flattened spatial tokens (Fig. 2)."""
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
        O = self.to_out(O)
        return (x_flat + O).permute(0, 2, 1).view(B, C, H, W, D)


class FeedForward3D(nn.Module):
    """Position-wise feed-forward network over spatial tokens (Fig. 2)."""
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
    """Self-attn + cross-attn + FFN — one transformer block from paper Fig. 2."""
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


class UNetBlock3D(nn.Module):
    """(ResBlock + 3×TransformerBlock) × 2 — paper Fig. 2 encoder/decoder block."""
    def __init__(self, in_ch, out_ch, t_dim, context_dim=COND_DIM, n_transformer=3,
                 downsample_ch=None, upsample_ch=None, upsample_in_ch=None):
        super().__init__()
        self.norm1   = nn.GroupNorm(GROUPNORM_GROUPS, in_ch, eps=1e-6)
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

    def forward(self, x, t_emb, ctx):
        h  = self.conv1(F.silu(self.norm1(x)))
        h  = h + self.t_proj1(F.silu(t_emb))[:, :, None, None, None]
        h  = self.conv2(F.silu(self.norm2(h))) + self.skip1(x)
        for blk in self.trans1:
            h = blk(h, ctx)
        r  = self.conv3(F.silu(self.norm3(h)))
        r  = r + self.t_proj2(F.silu(t_emb))[:, :, None, None, None]
        h  = h + self.conv4(F.silu(self.norm4(r)))
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


class DenoisingUNet3D(nn.Module):
    """Text-guided denoising U-Net (paper §II-D).

    3 resolution levels, channel widths [256, 512, 768], 2 residual blocks +
    6 transformer blocks per level.
    Input : h_t = cat(z_t, z_m)  shape (B, LATENT_CH*2, lH, lW, lD)
    Output: predicted noise        shape (B,   LATENT_CH, lH, lW, lD)
    """
    def __init__(self, in_ch=LATENT_CH * 2, ch_list=(256, 512, 768),
                 t_dim=256, context_dim=COND_DIM):
        super().__init__()
        c1, c2, c3 = ch_list

        self.t_embed = TimestepEmbedding(t_dim)
        self.in_conv = nn.Conv3d(in_ch, c1, 3, padding=1)

        self.enc1  = UNetBlock3D(c1, c1, t_dim, context_dim, downsample_ch=c2)
        self.enc2  = UNetBlock3D(c2, c2, t_dim, context_dim, downsample_ch=c3)
        self.enc3  = UNetBlock3D(c3, c3, t_dim, context_dim, downsample_ch=c3)

        self.mid   = UNetBlock3D(c3, c3, t_dim, context_dim)

        self.up3 = nn.ConvTranspose3d(c3, c3, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1)

        self.dec3  = UNetBlock3D(c3 * 2, c3, t_dim, context_dim)
        self.dec2  = UNetBlock3D(c2 * 2, c2, t_dim, context_dim)
        self.dec1  = UNetBlock3D(c1 * 2, c1, t_dim, context_dim)

        self.out   = nn.Sequential(
            nn.GroupNorm(GROUPNORM_GROUPS, c1, eps=1e-6), nn.SiLU(), nn.Conv3d(c1, LATENT_CH, 1)
        )

    def forward(self, ht, t, ctx):
        t_emb = self.t_embed(t)
        x  = self.in_conv(ht)

        e1, x2 = self.enc1(x,  t_emb, ctx)
        e2, x3 = self.enc2(x2, t_emb, ctx)
        e3, xm = self.enc3(x3, t_emb, ctx)

        _, m = self.mid(xm, t_emb, ctx)

        u3 = F.interpolate(self.up3(m),  size=e3.shape[2:], mode='trilinear', align_corners=False)
        u2 = F.interpolate(self.up2(u3), size=e2.shape[2:], mode='trilinear', align_corners=False)
        u1 = F.interpolate(self.up1(u2), size=e1.shape[2:], mode='trilinear', align_corners=False)

        _, d3 = self.dec3(torch.cat([u3, e3], dim=1), t_emb, ctx)
        _, d2 = self.dec2(torch.cat([u2, e2], dim=1), t_emb, ctx)
        _, d1 = self.dec1(torch.cat([u1, e1], dim=1), t_emb, ctx)

        return self.out(d1)


# ── CLIP Text Encoder ─────────────────────────────────────────────────────────

def load_clip_encoder(clip_model_path=CLIP_MODEL_PATH, device=DEVICE):
    """Load frozen CLIP text encoder. Returns (tokenizer, text_enc, clip_dim)."""
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path)
    clip_text_enc  = CLIPTextModel.from_pretrained(clip_model_path).to(device)
    clip_text_enc.eval()
    for p in clip_text_enc.parameters():
        p.requires_grad_(False)
    clip_dim = clip_text_enc.config.hidden_size
    print(f"CLIP hidden dim: {clip_dim}")
    return clip_tokenizer, clip_text_enc, clip_dim


def make_encode_atrophy(clip_tokenizer, clip_text_enc, device=DEVICE):
    @torch.no_grad()
    def encode_atrophy(atrophy_vals):
        prompts = []
        for i in range(atrophy_vals.shape[0]):
            scores     = atrophy_vals[i, :20].tolist()
            scores_str = " ".join([f"{v:.1f}" for v in scores])
            prompts.append(f"Atrophy: {scores_str}")
        tokens = clip_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        return clip_text_enc(**tokens).last_hidden_state
    return encode_atrophy


def make_encode_ptau(clip_tokenizer, clip_text_enc, device=DEVICE):
    @torch.no_grad()
    def encode_ptau(ptau_vals):
        prompts = [f"Plasma is {ptau_vals[i, 0].item():.3f}."
                   for i in range(ptau_vals.shape[0])]
        tokens = clip_tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        return clip_text_enc(**tokens).last_hidden_state
    return encode_ptau
