"""Small shared building blocks: padding helpers, GAN losses, image pool, weight init."""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to(x, target, value=0.0):
    """Symmetrically pad the spatial dims of x (N,C,*spatial) up to `target`.
    Returns (padded, pads) where pads is the F.pad spec (last dim first)."""
    spatial = x.shape[2:]
    pads = []
    for s, t in zip(reversed(spatial), reversed(target)):
        extra = max(t - s, 0)
        lo = extra // 2
        pads += [lo, extra - lo]
    if any(pads):
        x = F.pad(x, pads, mode="constant", value=value)
    return x, pads


def pad_to_multiple(x, m, value=0.0):
    target = [((s + m - 1) // m) * m for s in x.shape[2:]]
    return pad_to(x, target, value)


def crop_pads(y, pads):
    """Undo pad_to / pad_to_multiple (pads in F.pad order: last spatial dim first)."""
    if not any(pads):
        return y
    nd = len(pads) // 2
    slices = [slice(None), slice(None)]
    for d in range(nd):  # spatial dims in forward order
        lo, hi = pads[2 * (nd - 1 - d)], pads[2 * (nd - 1 - d) + 1]
        size = y.shape[2 + d]
        slices.append(slice(lo, size - hi))
    return y[tuple(slices)]


def to_pm1(x):
    """[0,1] → [-1,1]  (Tanh-output GANs; background 0 → -1)."""
    return x * 2.0 - 1.0


def from_pm1(x):
    return (x + 1.0) / 2.0


class GANLoss(nn.Module):
    """LSGAN (MSE) or vanilla (BCE-with-logits) adversarial loss on patch maps.

    label_smooth=True reproduces Ea-GAN's GANLoss_smooth: per call a ~ U(0,1) and the
    real target is 1 + 0.5a - 0.3 (in [0.7, 1.2]), the fake target 0.3a (in [0, 0.3]).
    """

    def __init__(self, mode="lsgan", label_smooth=False):
        super().__init__()
        assert mode in ("lsgan", "vanilla")
        self.mode = mode
        self.label_smooth = label_smooth
        self.loss = nn.MSELoss() if mode == "lsgan" else nn.BCEWithLogitsLoss()

    def forward(self, pred, target_is_real):
        if self.label_smooth:
            a = random.uniform(0, 1)
            val = (1.0 + a * 0.5 - 0.3) if target_is_real else (a * 0.3)
        else:
            val = 1.0 if target_is_real else 0.0
        target = torch.full_like(pred, val)
        return self.loss(pred, target)


class ImagePool:
    """pix2pix/CycleGAN history buffer of generated images for the discriminator."""

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size <= 0:
            return images
        out = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                out.append(img)
            elif random.uniform(0, 1) > 0.5:
                k = random.randint(0, self.pool_size - 1)
                tmp = self.images[k].clone()
                self.images[k] = img
                out.append(tmp)
            else:
                out.append(img)
        return torch.cat(out, 0)


def init_weights(net, init_type="normal", gain=0.02):
    """pix2pix-style init: conv/linear weights ~ N(0, gain) (or kaiming), norm weights ~ N(1, gain)."""

    def fn(m):
        name = m.__class__.__name__
        if hasattr(m, "weight") and m.weight is not None and ("Conv" in name or "Linear" in name):
            if init_type == "normal":
                nn.init.normal_(m.weight, 0.0, gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight, gain=gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)
        elif "BatchNorm" in name and getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight, 1.0, gain)
            nn.init.constant_(m.bias, 0.0)

    net.apply(fn)
    return net


def count_params(m):
    return sum(p.numel() for p in m.parameters())
