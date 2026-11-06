#!/bin/bash
#SBATCH --job-name=taugennet_large
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=results/logs/taugennet_large_%j.out
#SBATCH --error=results/logs/taugennet_large_%j.err

# Large architecture: ch_list=(256,512,768), n_transformer=3
# Baseline for model-size comparison vs train_small.slurm
# Uses existing shared AE checkpoint — no AE retraining.

PYTHON=/usr/bin/python3.9
AE_CKPT=results/checkpoints/taugennet_checkpoint_postact.pt

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

mkdir -p results/logs \
         results/checkpoints/large_arch \
         results/figures/large \
         results/figures/monitor_large

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Atrophy ──────────────────────────────────────────────────────────────────
echo "=== Atrophy (large arch) ==="
$PYTHON scripts/train.py \
    --mode atrophy \
    --skip-ae \
    --ae-checkpoint $AE_CKPT \
    --diff-epochs 2000 \
    --patience 100 \
    --val-every 5 \
    --monitor-every 25 \
    --unet-channels 256,512,768 \
    --n-transformer 3 \
    --checkpoint-dir results/checkpoints/large_arch \
    --monitor-dir results/figures/monitor_large \
    --figures-dir results/figures/large
echo "Atrophy done at: $(date)"
echo ""

# ── Combined ─────────────────────────────────────────────────────────────────
echo "=== Combined (large arch) ==="
$PYTHON scripts/train.py \
    --mode combined \
    --skip-ae \
    --ae-checkpoint $AE_CKPT \
    --diff-epochs 2000 \
    --patience 100 \
    --val-every 5 \
    --monitor-every 25 \
    --unet-channels 256,512,768 \
    --n-transformer 3 \
    --checkpoint-dir results/checkpoints/large_arch \
    --monitor-dir results/figures/monitor_large \
    --figures-dir results/figures/large
echo "Combined done at: $(date)"
echo ""

echo "All modes complete. End time: $(date)"
