# TauGenNet

A 3D diffusion model that generates tau PET brain scans conditioned on MRI, plus a
deterministic baseline it's evaluated against.

- **Input resolution:** 96×112×96
- **Sampler:** DDPM (500 steps), not DDIM
- **Conditioning:** MRI (structural) + optional demographic/biomarker variables (regional
  atrophy z-scores, plasma p-tau217)

## What's in this repo

| Folder | What it is |
|---|---|
| [`taugennet/`](taugennet/) | The diffusion model — training, inference, and conditioning code (`src/`, `scripts/`, `slurm/`). |
| [`baselines/tau-denseunet/`](baselines/tau-denseunet/) | A deterministic 3D Dense-U-Net baseline (MRI→tau-PET regression) it's compared against — a PyTorch port of [Neurology-AI-Program/AI_imputed_tau_PET](https://github.com/Neurology-AI-Program/AI_imputed_tau_PET) (Kolařík et al. 2019), retrained on the same lab data/split. |

The two models are trained and evaluated on the *same* cohort/split so their results are
directly comparable — see [`baselines/tau-denseunet/README.md`](baselines/tau-denseunet/README.md)
for exactly how it reuses this repo's data loader and evaluation pipeline.

## Evaluation

Both models are scored with `taugennet/scripts/evaluate_final.py` — no metric code is
duplicated for the baseline. It computes, in SUVR (unnormalized) space, DK86-masked:

1. **Pooled voxel-level, whole brain** — NRMSE / MAE / MSE / SSIM / Pearson r over all
   in-brain voxels of all subjects pooled.
2. **Pooled voxel-level, per region** (86 Desikan–Killiany regions).
3. **Regional cross-subject** — per-region subject-level means correlated real vs. generated
   across subjects.

Figures (hexbin, region bar charts, glass brain) come from
`taugennet/scripts/suvr_eval_figures.py`.

## Data

This repo does **not** include ADNI data or trained checkpoints — the ADNI Data Use
Agreement doesn't permit redistributing subject-level imaging or biomarker data. Both models
expect a `TAUGENNET_ROOT` environment variable pointing at a local `data/raw/` populated
per each subfolder's README.

## Environment

```bash
export TAUGENNET_ROOT=/path/to/this/repo/taugennet
export PYTHONPATH=$TAUGENNET_ROOT:${PYTHONPATH:-}
```

See [`taugennet/CLAUDE.md`](taugennet/CLAUDE.md) for the full pipeline (training launch,
evaluation launch, canonical metric sets) and [`baselines/tau-denseunet/README.md`](baselines/tau-denseunet/README.md)
for the baseline's usage.
