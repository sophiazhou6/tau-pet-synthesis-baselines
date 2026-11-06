# taugennet

A 3D diffusion model that synthesizes tau PET brain scans from structural MRI, optionally
conditioned on demographic/biomarker variables.

## What this is (and what it isn't)

This follows the architecture of **Gong et al., "TauGenNet: Plasma-Driven Tau PET Image
Synthesis via Text-Guided 3D Diffusion Models"** (*IEEE Transactions on Radiation and Plasma
Medical Sciences*; preprint [arXiv:2509.04269](https://arxiv.org/abs/2509.04269)) — a
DDPM-based (500-step) 3D diffusion model that generates tau PET conditioned on MRI, with the
plasma p-tau217 biomarker injected as text guidance.

This project extends it beyond a literal port: it adds an alternative conditioning path
(regional atrophy z-scores, both as an MLP vector and painted as a spatial map), an
architecture/hyperparameter sweep (latent channels, model width, noise schedule), 5-fold CV,
and DK86-masked regional evaluation not present in the original paper. The root
[README.md](../README.md) lists what each script does; `slurm/` holds the corresponding job
scripts for grid search, ablations, and conditioning variants.

- **Input resolution:** 96×112×96
- **Sampler:** DDPM (500 steps), not DDIM
- **Conditioning:** MRI (structural) + optional regional atrophy z-scores and/or plasma
  p-tau217 (text-guided)

## Layout

```
src/       model, diffusion schedule, dataset loaders, conditioning encoders
scripts/   train.py, evaluate_final.py, generation + figure scripts
slurm/     SLURM batch scripts (training, evaluation, grid search, ablations)
```

## Usage

See the root [README.md](../README.md) — Quickstart, plus a per-file description of everything
in `src/`, `scripts/`, and `slurm/`.
