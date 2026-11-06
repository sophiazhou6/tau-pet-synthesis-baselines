"""
diffusion.py – linear beta schedule, forward diffusion, and DDPM reverse step.

Extracted from models.py so it can be imported independently.
"""

import math

import torch
import torch.nn.functional as F

from .config import T_STEPS, BETA_START, BETA_END, DEVICE


class DiffusionSchedule:
    """Beta schedule supporting 'linear' (default) and 'cosine' (Nichol & Dhariwal 2021)."""

    def __init__(self, T=T_STEPS, beta_start=BETA_START, beta_end=BETA_END, device=DEVICE,
                 schedule="linear"):
        self.T      = T
        self.device = device
        self.schedule_name = schedule   # "linear" | "cosine"; persisted in checkpoints

        if schedule == "cosine":
            s = 0.008
            t = torch.arange(T + 1, dtype=torch.float32)
            f = torch.cos(((t / T) + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar_full = f / f[0]
            alpha_bar      = alpha_bar_full[1:]
            alpha_bar_prev = alpha_bar_full[:-1]
            betas          = (1 - alpha_bar / alpha_bar_prev).clamp(0, 0.999)
            alphas         = 1 - betas
        else:
            betas          = torch.linspace(beta_start, beta_end, T)
            alphas         = 1 - betas
            alpha_bar      = torch.cumprod(alphas, dim=0)
            alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        post_var = (betas * (1 - alpha_bar_prev) / (1 - alpha_bar)).clamp(min=1e-20)

        for name, val in [
            ("betas",         betas),
            ("alphas",        alphas),
            ("alpha_bar",     alpha_bar),
            ("alpha_bar_prev", alpha_bar_prev),
            ("sqrt_ab",       alpha_bar.sqrt()),
            ("sqrt_one_m_ab", (1 - alpha_bar).sqrt()),
            ("post_var",      post_var),
        ]:
            setattr(self, name, val.to(device))

    def q_sample(self, z0, t, noise=None):
        """Forward diffusion: z_t = sqrt(ā_t)·z_0 + sqrt(1−ā_t)·ε  (Eq. 5)."""
        if noise is None:
            noise = torch.randn_like(z0)
        s  = self.sqrt_ab[t].view(-1, 1, 1, 1, 1)
        sm = self.sqrt_one_m_ab[t].view(-1, 1, 1, 1, 1)
        return s * z0 + sm * noise, noise

    def p_sample_from_eps(self, zt, t_idx, eps_hat):
        """DDPM reverse step given a pre-computed eps (e.g. CFG-guided eps)."""
        z0_hat = (zt - self.sqrt_one_m_ab[t_idx] * eps_hat) / self.sqrt_ab[t_idx]
        beta_t = self.betas[t_idx]
        coef1  = self.alpha_bar_prev[t_idx].sqrt() * beta_t / (1 - self.alpha_bar[t_idx])
        coef2  = self.alphas[t_idx].sqrt() * (1 - self.alpha_bar_prev[t_idx]) / (1 - self.alpha_bar[t_idx])
        mean   = coef1 * z0_hat + coef2 * zt
        if t_idx == 0:
            return mean
        return mean + self.post_var[t_idx].sqrt() * torch.randn_like(zt)

    @torch.no_grad()
    def p_sample(self, model, zt, t_idx, zm, ctx):
        """One DDPM reverse step: z_{t-1} ~ p_θ(z_{t-1} | z_t).

        NOTE: only correct when called with consecutive t values (t, t-1).
        Use ddim_sample for subsampled inference (skipping steps).
        """
        t_batch = torch.full((zt.shape[0],), t_idx, device=self.device, dtype=torch.long)
        ht      = torch.cat([zt, zm], dim=1)
        eps_hat = model(ht, t_batch, ctx)

        # Reconstruct z0 estimate
        z0_hat = (zt - self.sqrt_one_m_ab[t_idx] * eps_hat) / self.sqrt_ab[t_idx]

        # Posterior mean coefficients (DDPM Eq. 7)
        beta_t = self.betas[t_idx]
        coef1  = self.alpha_bar_prev[t_idx].sqrt() * beta_t / (1 - self.alpha_bar[t_idx])
        coef2  = self.alphas[t_idx].sqrt() * (1 - self.alpha_bar_prev[t_idx]) / (1 - self.alpha_bar[t_idx])
        mean   = coef1 * z0_hat + coef2 * zt

        if t_idx == 0:
            return mean
        return mean + self.post_var[t_idx].sqrt() * torch.randn_like(zt)

    @torch.no_grad()
    def ddim_sample(self, model, zt, t_idx, t_prev_idx, zm, ctx):
        """One DDIM reverse step — correct for any step size (Song et al. 2020).

        t_prev_idx: the actual previous timestep in the subsampled schedule,
                    e.g. t_idx=980, t_prev_idx=960 for 50-step inference.
                    Pass -1 (or 0) for the last step where prev alpha_bar = 1.
        """
        t_batch = torch.full((zt.shape[0],), t_idx, device=self.device, dtype=torch.long)
        ht      = torch.cat([zt, zm], dim=1)
        eps_hat = model(ht, t_batch, ctx)

        # Predicted x0
        z0_hat = (zt - self.sqrt_one_m_ab[t_idx] * eps_hat) / self.sqrt_ab[t_idx]

        # alpha_bar at the actual previous step in the subsequence
        ab_prev = self.alpha_bar[t_prev_idx] if t_prev_idx >= 0 else torch.tensor(1.0, device=self.device)

        # DDIM update (deterministic: eta=0)
        return ab_prev.sqrt() * z0_hat + (1 - ab_prev).sqrt() * eps_hat
