#!/bin/bash
#SBATCH --job-name=canary_spatial_B
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/canary_spatial_B_%j.out
#SBATCH --error=results/logs/canary_spatial_B_%j.err

# 2-epoch smoke test for spatial-injection Design B (learned AtrophyEncoder3D).
# Catches shape/construction bugs fast (~15min walltime backfills faster than a
# full job on a contended queue). NOT a real training run — --diff-epochs 2
# --patience 1 only exercise: dataset build, latent caching, one train epoch,
# one val pass (val-every 1), one _monitor_ssim pass (monitor-every 1),
# checkpoint save+reload code paths.
#
# CoreHPC env confirmed 2026-07-14: python3.9 (no conda), gpu:nvidia_h100_nvl:1
# on partition=gpu, HOME=/home/remote/adebnath1.
# AE checkpoint: paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt does NOT match
# this repo's current Encoder3D/Decoder3D (confirmed via load_state_dict — channel-width
# and layer-count mismatches, not just latent_ch) — empirically verified 2026-07-15 that
# ae_lch8_dk86/ae_checkpoint_best.pt is the only AE checkpoint on this cluster that
# actually loads against the current Autoencoder3D, with latent_ch=8.
# NOTE: the doc's launch reference used `--cond-mode ptau217_mlp --cfg-prob 0.15`,
# but this script's actual argparse has no such mode/flag (--cond-mode choices are
# atrophy|none|ptau217, no --cfg-prob) — dropped here to avoid an argparse crash;
# use --cond-mode ptau217 if that's the intended cross-attention conditioner.

set -e   # fail loudly (nonzero sacct exit) instead of masking a crash with the trailing echo

PYTHON=/usr/bin/python3.9
cd $HOME/taugennet
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:$PYTHONPATH
export TAUGENNET_ROOT=$HOME/taugennet

AE_CKPT="results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt"
OUTDIR="results/checkpoints/spatial_atrophy/canary_B"
FIGDIR="results/figures/spatial_atrophy/canary_B"
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
    --checkpoint-dir "${OUTDIR}" \
    --figures-dir "${FIGDIR}" \
    --use-mask

echo "Done: $(date)"
