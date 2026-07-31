# tau-denseunet-baseline

A PyTorch **baseline** for TauGenNet: a deterministic 3D **Dense-U-Net** that
synthesizes tau PET from MRI, trained and evaluated on the same lab data and split
as the TauGenNet diffusion model so the two can be compared head-to-head.

## What this is (and what it isn't)

This is a faithful port of the model in
[Neurology-AI-Program/AI_imputed_tau_PET](https://github.com/Neurology-AI-Program/AI_imputed_tau_PET)
(`tau_synthesis_train.py:get_unet`), originally the 3D Dense-U-Net of
Kolařík et al. (2019). The upstream repo is TensorFlow/Keras and performs
**FDG PET → tau PET** regression at 128³, reading pre-baked HDF5 `.mat` volumes.

This port differs deliberately:

| | Upstream (Keras) | This baseline (PyTorch) |
|---|---|---|
| Framework | TensorFlow/Keras | PyTorch |
| Input → output | FDG PET → tau PET | **MRI → tau PET** |
| Resolution | 128³ | 96×112×96 (lab `VOL_SHAPE`) |
| Data | HDF5 `.mat`, k-fold | TauGenNet lab pipeline, TauGenNet split |
| Eval | external | TauGenNet `evaluate_final.py` |

**Why MRI→tau, not FDG→tau:** the lab data on disk has only MRI (`T1_to_MNI_nonlin`)
and tau PET (cerebellum-normalized SUVR) — no preprocessed FDG. MRI→tau also matches
exactly what TauGenNet conditions on, which is what makes this a clean baseline. The
model is single-input-channel, so FDG can be swapped in later if the lab provides it.

The architecture (dense skip pattern, channel counts, linear output, MSE loss) is
reproduced verbatim from upstream; see [`denseunet/model.py`](denseunet/model.py)
and the vendored originals in [`docs/`](docs/).

## Data — not copied, reused in place

This repo does **not** contain or copy the ADNI raw data. It imports TauGenNet's
data loader (`src.dataset_final.build_dataloaders`), which reads from
`$TAUGENNET_ROOT/data/raw`. There is one source of truth for the data
(`/scratch/network/sz3962/taugennet/data/raw`); set `TAUGENNET_ROOT` if it moves.

## Environment

Reuses the existing `taugennet` conda env (PyTorch, nibabel, pandas, nilearn):

```bash
PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
```

## Usage

```bash
export TAUGENNET_ROOT=/scratch/network/sz3962/taugennet

# Train + cache test predictions for one mode (atrophy | ptau217)
sbatch slurm/train_generate.slurm atrophy

# Or interactively, step by step:
$PYTHON scripts/train.py    --mode atrophy
$PYTHON scripts/generate.py --mode atrophy   # writes results/generated/atrophy/subject_*.npy
```

`--mode` selects the matched-subject cohort and split (the conditioning vector
itself is unused — this is an image-to-image model). Use the same mode you compare
against in TauGenNet.

## Evaluation — reuses TauGenNet's canonical pipeline

`generate.py` writes `subject_{i:03d}.npy` in normalized [0,1] space, indexed to
match the test-set ordering of `build_dataloaders(mode=...)` — exactly the cache
format `evaluate_final.py` consumes. So scoring is identical to TauGenNet:

```bash
cd $TAUGENNET_ROOT
$PYTHON scripts/evaluate_final.py --mode atrophy --use-cached \
    --generated-dir /scratch/network/sz3962/tau-denseunet-baseline/results/generated/atrophy
```

This produces the full DK86-masked metric suite (NRMSE, MAE, MSE, SSIM, Pearson;
pooled whole-brain, per-region, and regional cross-subject) in the same tables used
for TauGenNet — no metric code is duplicated here.

## Layout

```
denseunet/model.py     3D Dense-U-Net (PyTorch port)
denseunet/config.py    paths + reused lab data loader / VOL_SHAPE / split
scripts/train.py       train (Adam + masked MSE), early stopping, no-overwrite ckpts
scripts/generate.py    cache test predictions for evaluate_final.py
slurm/train_generate.slurm
docs/                  vendored upstream reference (Keras train/test, cohort CSV)
```
