#!/bin/bash
#SBATCH --job-name=taugennet_small
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=results/logs/taugennet_small_%j.out
#SBATCH --error=results/logs/taugennet_small_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Small architecture: ch_list=(128,256,512), n_transformer=1
# Reduced-capacity model for model-size comparison vs train_large.slurm
# Uses existing shared AE checkpoint — no AE retraining.

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
AE_CKPT=results/checkpoints/taugennet_checkpoint_postact.pt

cd /scratch/network/sz3962/taugennet

mkdir -p results/logs \
         results/checkpoints/small_arch \
         results/figures/small \
         results/figures/monitor_small

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Atrophy ──────────────────────────────────────────────────────────────────
echo "=== Atrophy (small arch) ==="
$PYTHON scripts/train.py \
    --mode atrophy \
    --skip-ae \
    --ae-checkpoint $AE_CKPT \
    --diff-epochs 2000 \
    --patience 100 \
    --val-every 5 \
    --monitor-every 25 \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --checkpoint-dir results/checkpoints/small_arch \
    --monitor-dir results/figures/monitor_small \
    --figures-dir results/figures/small
echo "Atrophy done at: $(date)"
echo ""

# ── Combined ─────────────────────────────────────────────────────────────────
echo "=== Combined (small arch) ==="
$PYTHON scripts/train.py \
    --mode combined \
    --skip-ae \
    --ae-checkpoint $AE_CKPT \
    --diff-epochs 2000 \
    --patience 100 \
    --val-every 5 \
    --monitor-every 25 \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --checkpoint-dir results/checkpoints/small_arch \
    --monitor-dir results/figures/monitor_small \
    --figures-dir results/figures/small
echo "Combined done at: $(date)"
echo ""

echo "All modes complete. End time: $(date)"
