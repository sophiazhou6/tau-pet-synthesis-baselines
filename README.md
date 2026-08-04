# tau-pet-synthesis-baselines

Two competing published baselines for MRI→tau-PET synthesis, reimplemented and evaluated
head-to-head on the same ADNI cohort/split. This is a baseline-comparison study — a reference
point for other tau-PET-synthesis modeling work, not a novel model in its own right.

## What's in this repo

| Folder | What it is |
|---|---|
| [`baselines/taugennet/`](baselines/taugennet/) | A 3D MRI-conditioned diffusion model (DDPM, 500 steps), following Gong et al., ["TauGenNet: Plasma-Driven Tau PET Image Synthesis via Text-Guided 3D Diffusion Models"](https://arxiv.org/abs/2509.04269) (IEEE Trans. Radiation and Plasma Medical Sciences), extended with an additional atrophy-based conditioning path and an architecture sweep. |
| [`baselines/tau-denseunet/`](baselines/tau-denseunet/) | A deterministic 3D Dense-U-Net (MRI→tau-PET regression) — a PyTorch port of [Neurology-AI-Program/AI_imputed_tau_PET](https://github.com/Neurology-AI-Program/AI_imputed_tau_PET) (Kolařík et al. 2019). |

Both are trained and evaluated on the *same* cohort/split so their results are directly
comparable — see each folder's README for what was ported verbatim vs. adapted from its
source paper.

## Evaluation

Both models are scored with `baselines/taugennet/scripts/evaluate_final.py` — one shared
pipeline, no metric code duplicated between them. It computes, in SUVR (unnormalized) space,
DK86-masked:

1. **Pooled voxel-level, whole brain** — NRMSE / MAE / MSE / SSIM / Pearson r over all
   in-brain voxels of all subjects pooled.
2. **Pooled voxel-level, per region** (86 Desikan–Killiany regions).
3. **Regional cross-subject** — per-region subject-level means correlated real vs. generated
   across subjects.

Figures (hexbin, region bar charts, glass brain) come from
`baselines/taugennet/scripts/suvr_eval_figures.py`.

## Data

This repo does **not** include ADNI data or trained checkpoints — the ADNI Data Use
Agreement doesn't permit redistributing subject-level imaging or biomarker data. Both models
expect a `TAUGENNET_ROOT` environment variable pointing at a local `data/raw/` populated
with your own ADNI tau-PET + MRI pull (see `baselines/taugennet/src/config.py` and
`baselines/taugennet/src/dataset_final.py` for the exact expected file layout).

## Requirements

Developed against Python 3.9. Third-party deps (no `requirements.txt` yet — install via
conda/pip as you prefer): `torch`, `nibabel`, `numpy`, `scipy`, `scikit-image`, `nilearn`,
`matplotlib`, `pandas`, `tqdm`, `transformers` (the last only needed for the `ptau217`/
`combined` conditioning modes, which use a frozen CLIP/BioBERT text encoder).

## Environment

```bash
export TAUGENNET_ROOT=/path/to/this/repo/baselines/taugennet
export PYTHONPATH=$TAUGENNET_ROOT:${PYTHONPATH:-}
```

`TAUGENNET_ROOT` is read by `src/config.py` — if it's unset, paths silently fall back to a
stale hardcoded default and everything writes into the wrong place, so always set it first.

## Quickstart — diffusion baseline (`baselines/taugennet/`)

```bash
cd baselines/taugennet

# Train (AE + diffusion, mentor split, default 64/16/20) — `--mode` is the only required flag
python scripts/train.py --mode atrophy

# Evaluate: run fresh inference from a checkpoint, cache the generations, and score them
python scripts/evaluate_final.py --mode atrophy \
    --checkpoint-dir results/checkpoints/<your_run> \
    --save-generated --generated-dir results/generated/atrophy

# Re-score cached generations later without a GPU / re-running inference
python scripts/evaluate_final.py --mode atrophy --use-cached \
    --generated-dir results/generated/atrophy
```

`--mode` selects the conditioning: `atrophy` (regional atrophy z-scores), `ptau217`
(plasma biomarker via text encoder), or `combined`. `evaluate_final.py` writes the full
DK86-masked metric suite (see Evaluation above) plus glass-brain figures by default. SLURM
batch scripts for both training and evaluation are under `slurm/` if you have access to a
SLURM cluster; see [`baselines/taugennet/CLAUDE.md`](baselines/taugennet/CLAUDE.md) for the
full set of launch variants (grid search, ablations, CV folds) and cluster-specific notes.

## Quickstart — DenseUNet baseline (`baselines/tau-denseunet/`)

```bash
cd baselines/tau-denseunet

python scripts/train.py    --mode atrophy
python scripts/generate.py --mode atrophy   # writes results/generated/atrophy/subject_*.npy

# Score with the shared evaluate_final.py — same pipeline, no duplicated metric code
cd $TAUGENNET_ROOT
python scripts/evaluate_final.py --mode atrophy --use-cached \
    --generated-dir <path-to>/baselines/tau-denseunet/results/generated/atrophy
```

See [`baselines/tau-denseunet/README.md`](baselines/tau-denseunet/README.md) for the full
usage notes and what this baseline is (and isn't) porting from upstream.
