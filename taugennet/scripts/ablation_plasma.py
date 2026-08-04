#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""Conditioning ablation for the COMBINED model (plasma-shortcut test).

Combined conditioning = [atrophy z-scores (86) | plasma p-tau217 (1)].  This script
regenerates the test set under three conditions, from the SAME checkpoint and the SAME
DDPM-500 pipeline as the baseline, differing only in the conditioning vector:

  full        — unchanged conditioning (baseline; regenerated for a fair comparison)
  no_plasma   — plasma element zeroed   (cond[-1] = 0)  → tests if spatial M2 recovers
  no_atrophy  — atrophy block zeroed    (cond[:86] = 0) → tests if burden M1 survives

Outputs are written ONLY to:
    results/generated/<arch>/<mode>_ablation/<condition>/subject_{i:03d}.npy
so nothing existing is touched.  Afterwards score each with the 2×2:
    python scripts/crosssubject_2x2.py --mode combined --arch silu \
        --generated-dir results/generated/silu_500ep/combined_ablation/no_plasma

Usage:
    python scripts/ablation_plasma.py \
        --checkpoint results/checkpoints/silu/combined_500ep/diff_combined_best.pt \
        --out-base results/generated/silu_500ep/combined_ablation
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEVICE
from src.diffusion import DiffusionSchedule
from src.inference import load_models, synthesize_tau_pet
import src.dataset_combined as _dataset_combined
import src.dataset_combined_relu as _dataset_combined_relu


def ablate(cond, condition, mode):
    """Return a copy of the 1-D cond tensor with the requested block zeroed."""
    c = cond.clone()
    if condition == "full":
        return c
    if mode == "combined":
        if condition == "no_plasma":
            c[-1] = 0.0                 # plasma p-tau217 is the last element
        elif condition == "no_atrophy":
            c[:86] = 0.0                # 86 atrophy z-scores
    elif mode == "ptau217" and condition == "no_plasma":
        c[0] = 0.0
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="combined", choices=["combined", "ptau217"])
    p.add_argument("--arch", default="silu", choices=["silu", "relu"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-base", required=True,
                   help="Base dir; per-condition subdirs are created under it.")
    p.add_argument("--conditions", nargs="+",
                   default=["full", "no_plasma", "no_atrophy"])
    p.add_argument("--n-steps", type=int, default=500)
    p.add_argument("--max-subjects", type=int, default=None,
                   help="Limit subjects (smoke test).")
    args = p.parse_args()

    if args.mode == "ptau217" and "no_atrophy" in args.conditions:
        args.conditions = [c for c in args.conditions if c != "no_atrophy"]
        print("ptau217 has no atrophy block — dropping no_atrophy condition.")

    builder = _dataset_combined_relu if args.arch == "relu" else _dataset_combined
    _, _, test_ds, *_ = builder.build_dataloaders(mode=args.mode, use_dk_mask=True)
    n = len(test_ds) if args.max_subjects is None else min(args.max_subjects, len(test_ds))
    print(f"Test subjects: {len(test_ds)} (generating {n})")

    ae, unet, conditioner, latent_std, _ = load_models(
        args.checkpoint, args.mode, device=DEVICE, arch=args.arch)
    schedule    = DiffusionSchedule(device=DEVICE)
    encode_cond = conditioner.encode

    for cond_name in args.conditions:
        out_dir = os.path.join(args.out_base, cond_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== condition: {cond_name}  →  {out_dir} ===")
        for i in range(n):
            out_path = os.path.join(out_dir, f"subject_{i:03d}.npy")
            if os.path.exists(out_path):                      # resumable
                continue
            _, mri, cond = test_ds[i]
            cond_mod = ablate(cond, cond_name, args.mode)
            with torch.no_grad():
                gen = synthesize_tau_pet(
                    mri.unsqueeze(0).to(DEVICE),
                    cond_mod.unsqueeze(0).to(DEVICE),
                    ae, unet, schedule, encode_cond, latent_std,
                    n_steps=args.n_steps, sampler="ddpm",
                ).squeeze().cpu().numpy()
            np.save(out_path, gen)
            if (i + 1) % 5 == 0 or (i + 1) == n:
                print(f"  {cond_name}: {i+1}/{n}")
    print("\nAblation generation complete.")


if __name__ == "__main__":
    main()
