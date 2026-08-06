import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LATENT_CH, LATENT_SCALE, COND_DIM, DEVICE, GROUPNORM_GROUPS


# ── 3D Autoencoder ────────────────────────────────────────────────────────────
# Shared encoder/decoder for tau PET and MRI (single-channel), paper Sec. II-A (Eq. 1-3).
# PET and MRI use the same network weights to minimise parameters and ensure
# both modalities are compressed into the same latent space.

class ResBlock3D(nn.Module):
    # post-activation residual block: Conv → GN → SiLU
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
    # channel progression: 1 → ch_mult[0] → ch_mult[1] → ch_mult[2] → latent_ch * 2
    # ch_mult=(64, 128, 128) matches paper's [64, 128, 128, 128] adapted to 3 levels
    # (was base_ch=32 with doubling: [32, 64, 128, 256])
    def __init__(self, in_ch=1, ch_mult=(64, 128, 128), latent_ch=LATENT_CH, scale=LATENT_SCALE,
                 n_res_blocks=1):
        super().__init__()
        n_down = int(math.log2(scale))  # 3 for scale=8
        assert len(ch_mult) == n_down, f"ch_mult must have {n_down} entries for scale={scale}"

        # Initial projection: 1 → ch_mult[0]
        layers = [nn.Conv3d(in_ch, ch_mult[0], 3, padding=1)]

        # Downsampling stages: n_res_blocks ResBlocks at current channels, then stride-2 conv.
        # n_res_blocks=1 reproduces the original per-stage layout (depth axis of the grid).
        for i in range(n_down):
            ch      = ch_mult[i]
            ch_next = ch_mult[i + 1] if i + 1 < n_down else ch  # last stage stays at ch_mult[-1]
            layers += [ResBlock3D(ch) for _ in range(n_res_blocks)]
            layers += [nn.Conv3d(ch, ch_next, 4, stride=2, padding=1)]

        # Final (bottleneck) level: n_res_blocks ResBlocks + GroupNorm + SiLU + 1×1 conv to
        # latent mean/logvar. Counting the bottleneck, the encoder has n_down+1 levels, each with
        # n_res_blocks blocks -> n_res_blocks=2 gives the paper's "four levels, two residual blocks each".
        ch = ch_mult[-1]
        layers += [ResBlock3D(ch) for _ in range(n_res_blocks)]
        layers += [nn.GroupNorm(GROUPNORM_GROUPS, ch, eps=1e-6), nn.SiLU(),
                   nn.Conv3d(ch, latent_ch * 2, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        h = self.net(x)
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar


class Decoder3D(nn.Module):
    # mirror of Encoder3D; channel progression: latent_ch → ch_mult_rev → 1
    # (was base_ch=32 with reverse-doubling starting from 256)
    def __init__(self, latent_ch=LATENT_CH, ch_mult=(64, 128, 128), out_ch=1, scale=LATENT_SCALE,
                 n_res_blocks=1):
        super().__init__()
        n_up = int(math.log2(scale))  # 3
        ch_rev = list(reversed(ch_mult))  # (128, 128, 64)

        # Input projection from latent to highest channel count (bottleneck level: n_res_blocks blocks)
        layers = [nn.Conv3d(latent_ch, ch_rev[0], 3, padding=1)]
        layers += [ResBlock3D(ch_rev[0]) for _ in range(n_res_blocks)]

        # Upsampling stages: stride-2 ConvTranspose, then n_res_blocks ResBlocks.
        # n_res_blocks=1 reproduces the original per-stage layout (mirror of the encoder depth axis).
        for i in range(n_up):
            ch      = ch_rev[i]
            ch_next = ch_rev[i + 1] if i + 1 < n_up else ch_rev[-1]
            layers += [nn.ConvTranspose3d(ch, ch_next, 4, stride=2, padding=1)]
            layers += [ResBlock3D(ch_next) for _ in range(n_res_blocks)]

        ch_final = ch_rev[-1]  # 64
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

    def __init__(self, latent_ch=LATENT_CH, scale=LATENT_SCALE, n_res_blocks=1, ch_mult=None):
        super().__init__()
        self.latent_ch    = latent_ch
        self.scale        = scale         # latent spatial downsampling (grid: 4/8/16 -> finer/coarser latent)
        self.n_res_blocks = n_res_blocks  # AE depth: resblocks per stage (grid: 1/2/3)
        # Derive ch_mult to match `scale` if not given: 64, then 128 per extra downsampling stage
        # (scale=8 -> (64,128,128), the original default). All entries are /32 for GroupNorm.
        n_down = int(math.log2(scale))
        if ch_mult is None:
            ch_mult = tuple([64] + [128] * (n_down - 1))
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
        """Deterministic encoding — returns the distribution mean, no sampling."""
        mean, _ = self.encoder(x)
        return mean

    def decode(self, z): return self.decoder(z)

    def forward(self, x):
        z, mean, logvar = self.encode(x)
        return self.decode(z), mean, logvar


# ── Denoising U-Net with Cross-Attention ─────────────────────────────────────
# 3D U-Net accepting h_t = [z_t, z_m] with layer-wise cross-attention on CLIP
# embeddings (paper §II-D, Eq. 8-12).

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
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
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
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
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
    """(ResBlock + 3×TransformerBlock) × 2 — paper Fig. 2 encoder/decoder block.

    Two residual blocks each followed by three transformer blocks (self-attn +
    cross-attn + FFN), giving six cross-attention layers per UNet level.
    Channel mismatch (in_ch → out_ch) is handled in the first ResBlock's skip conv.
    At the bottom of the block (paper Fig. 2), optional downsample/upsample is applied.
    Returns (skip, out): skip = pre-downsample/upsample (for skip connections),
                        out  = post-downsample/upsample (passed to next level)
    (was: 2 conv layers + 1 cross-attention block)
    """
    def __init__(self, in_ch, out_ch, t_dim, context_dim=COND_DIM, n_transformer=3,
                 downsample_ch=None, upsample_ch=None, upsample_in_ch=None):
        super().__init__()
        # First ResBlock (handles channel change via skip)
        # eps=1e-6 per paper spec (PyTorch default was 1e-5)
        self.norm1   = nn.GroupNorm(GROUPNORM_GROUPS, out_ch, eps=1e-6)
        self.conv1   = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.t_proj1 = nn.Linear(t_dim, out_ch)
        self.norm2   = nn.GroupNorm(GROUPNORM_GROUPS, out_ch, eps=1e-6)
        self.conv2   = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip1   = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # Three transformer blocks after first ResBlock
        self.trans1  = nn.ModuleList([
            TransformerBlock3D(out_ch, context_dim) for _ in range(n_transformer)
        ])
        # Second ResBlock (same channels: out_ch → out_ch)
        self.norm3   = nn.GroupNorm(GROUPNORM_GROUPS, out_ch, eps=1e-6)
        self.conv3   = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.t_proj2 = nn.Linear(t_dim, out_ch)
        self.norm4   = nn.GroupNorm(GROUPNORM_GROUPS, out_ch, eps=1e-6)
        self.conv4   = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        # Three transformer blocks after second ResBlock
        self.trans2  = nn.ModuleList([
            TransformerBlock3D(out_ch, context_dim) for _ in range(n_transformer)
        ])
        # At the bottom of the block (paper Fig. 2)
        self.downsample = nn.Conv3d(out_ch, downsample_ch, 4, stride=2, padding=1) \
                          if downsample_ch is not None else None
        # upsample_in_ch: when set, upsample operates on the first upsample_in_ch
        # channels of the block INPUT (x) rather than the block output (h).
        # This matches the training configuration for dec2, where the upsample
        # was applied to up3 (768ch) passed in as the first part of the cat input.
        _up_in = upsample_in_ch if upsample_in_ch is not None else out_ch
        self.upsample        = nn.ConvTranspose3d(_up_in, upsample_ch, 4, stride=2, padding=1) \
                               if upsample_ch is not None else None
        self._upsample_in_ch = upsample_in_ch  # None → apply to h; int → apply to x[:, :n]

    def forward(self, x, t_emb, ctx):
        # First ResBlock (post-activation: Conv → GN → SiLU)
        h  = F.silu(self.norm1(self.conv1(x)))
        h  = h + self.t_proj1(F.silu(t_emb))[:, :, None, None, None]
        h  = F.silu(self.norm2(self.conv2(h))) + self.skip1(x)
        for blk in self.trans1:
            h = blk(h, ctx)
        # Second ResBlock (post-activation: Conv → GN → SiLU)
        r  = F.silu(self.norm3(self.conv3(h)))
        r  = r + self.t_proj2(F.silu(t_emb))[:, :, None, None, None]
        h  = h + F.silu(self.norm4(self.conv4(r)))
        for blk in self.trans2:
            h = blk(h, ctx)
        # h is now the skip output (pre-downsample/upsample)
        skip = h
        # Apply spatial resolution change at the bottom
        if self.downsample is not None:
            return skip, self.downsample(h)
        if self.upsample is not None:
            if self._upsample_in_ch is not None:
                return skip, self.upsample(x[:, :self._upsample_in_ch])
            return skip, self.upsample(h)
        return skip, h  # bottleneck — no spatial change


class DenoisingUNet3D(nn.Module):
    """Text-guided denoising U-Net (paper §II-D).

    3 resolution levels, channel widths [256, 512, 768], 2 residual blocks +
    6 transformer blocks per level.  (was: 2 levels [64, 128, 256], 1 attn/level)
    Downsample/upsample operations are now inside each UNetBlock3D at the bottom,
    matching paper Fig. 2 specification.

    Input : h_t = cat(z_t, z_m)  shape (B, LATENT_CH*2, lH, lW, lD)
    Output: predicted noise        shape (B,   LATENT_CH, lH, lW, lD)
    context_dim must match the output dimension of the conditioner used.
    """
    def __init__(self, latent_ch=LATENT_CH, in_ch=None, ch_list=(256, 512, 768),
                 t_dim=256, context_dim=COND_DIM, n_transformer=3):
        super().__init__()
        if in_ch is None:
            in_ch = latent_ch * 2
        c1, c2, c3 = ch_list
        nt = n_transformer

        self.t_embed = TimestepEmbedding(t_dim)
        self.in_conv = nn.Conv3d(in_ch, c1, 3, padding=1)

        # Encoder — downsample operations inside each block
        self.enc1  = UNetBlock3D(c1, c1, t_dim, context_dim, nt, downsample_ch=c2)
        self.enc2  = UNetBlock3D(c2, c2, t_dim, context_dim, nt, downsample_ch=c3)
        self.enc3  = UNetBlock3D(c3, c3, t_dim, context_dim, nt, downsample_ch=c3)

        # Bottleneck — no spatial change
        self.mid   = UNetBlock3D(c3, c3, t_dim, context_dim, nt)

        self.up3 = nn.ConvTranspose3d(c3, c3, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1)
        self.up1 = nn.ConvTranspose3d(c2, c1, 4, stride=2, padding=1)

        # Decoder: each block takes cat([u_n, enc_skip]) — no internal upsample
        self.dec3  = UNetBlock3D(c3 * 2, c3, t_dim, context_dim, nt)
        self.dec2  = UNetBlock3D(c2 * 2, c2, t_dim, context_dim, nt)
        self.dec1  = UNetBlock3D(c1 * 2, c1, t_dim, context_dim, nt)

        self.out   = nn.Sequential(
            nn.GroupNorm(GROUPNORM_GROUPS, c1, eps=1e-6), nn.SiLU(), nn.Conv3d(c1, latent_ch, 1)
        )

    def forward(self, ht, t, ctx):
        t_emb = self.t_embed(t)
        x  = self.in_conv(ht)

        # Encoder: unpack (skip, downsampled_out) from each block
        e1, x2 = self.enc1(x,  t_emb, ctx)   # e1=skip (pre-downsample), x2=downsampled
        e2, x3 = self.enc2(x2, t_emb, ctx)   # e2=skip (pre-downsample), x3=downsampled
        e3, xm = self.enc3(x3, t_emb, ctx)   # e3=skip (pre-downsample), xm=downsampled

        # Bottleneck
        _, m = self.mid(xm, t_emb, ctx)      # skip unused at bottleneck

        # Standalone upsample chain from bottleneck (up3→up2→up1)
        u3 = F.interpolate(self.up3(m),  size=e3.shape[2:], mode='trilinear', align_corners=False)
        u2 = F.interpolate(self.up2(u3), size=e2.shape[2:], mode='trilinear', align_corners=False)
        u1 = F.interpolate(self.up1(u2), size=e1.shape[2:], mode='trilinear', align_corners=False)

        # Decoder with encoder skip connections
        _, d3 = self.dec3(torch.cat([u3, e3], dim=1), t_emb, ctx)
        _, d2 = self.dec2(torch.cat([u2, e2], dim=1), t_emb, ctx)
        _, d1 = self.dec1(torch.cat([u1, e1], dim=1), t_emb, ctx)

        return self.out(d1)


if __name__ == "__main__":
    from .diffusion import DiffusionSchedule

    ae   = Autoencoder3D().to(DEVICE)
    unet = DenoisingUNet3D().to(DEVICE)
    print(f"Autoencoder parameters:     {sum(p.numel() for p in ae.parameters()):,}")
    print(f"Denoising U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")

    schedule = DiffusionSchedule()
    print(f"Diffusion schedule: T={schedule.T} steps")
