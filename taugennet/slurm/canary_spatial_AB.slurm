#!/bin/bash
#SBATCH --job-name=canary_spatial_AB
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/canary_spatial_AB_%j.out
#SBATCH --error=results/logs/canary_spatial_AB_%j.err

# 2-epoch smoke test for Design A + Design B combined (learned AtrophyEncoder3D
# feeding the single input-concat injection, PLUS SPADE with the raw map at every
# level). Only run this after B and A have each individually passed their own
# canary. CoreHPC env confirmed 2026-07-14 — see canary_spatial_B.slurm for details,
# including why AE_CKPT is ae_lch8_dk86 (paper_aligned/ae/fold_0 does not match the
# current Autoencoder3D architecture).

set -e   # fail loudly (nonzero sacct exit) instead of masking a crash with the trailing echo

PYTHON=/usr/bin/python3.9
cd $HOME/taugennet
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:$PYTHONPATH
export TAUGENNET_ROOT=$HOME/taugennet

AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
OUTDIR="results/checkpoints/spatial_atrophy/canary_AB"
FIGDIR="results/figures/spatial_atrophy/canary_AB"
mkdir -p "$OUTDIR" "$FIGDIR" results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON scripts/train_spatial_atrophy.py \
    --skip-ae \
    --ae-checkpoint "${AE_CKPT}" \
    --latent-ch 8 \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --diff-epochs 2 \
    --patience 1 \
    --val-every 1 \
    --monitor-every 1 \
    --use-atrophy-encoder \
    --atrophy-encoder-width 32 \
    --use-spade \
    --spade-hidden 64 \
    --checkpoint-dir "${OUTDIR}" \
    --figures-dir "${FIGDIR}" \
    --use-mask

echo "Done: $(date)"
