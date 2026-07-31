#!/bin/bash
#SBATCH --job-name=silu_500ep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/silu_500ep_%j.out
#SBATCH --error=results/logs/silu_500ep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# SiLU architecture, 500-epoch hard stop (patience=600 disables early stopping).
# Reuses the pretrained AE from the masked SiLU run.

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
AE_CKPT=results/checkpoints/taugennet_checkpoint_silu_masked.pt

cd /scratch/network/sz3962/taugennet

mkdir -p results/logs \
         results/checkpoints/silu/atrophy_500ep \
         results/checkpoints/silu/ptau217_500ep \
         results/checkpoints/silu/combined_500ep \
         results/figures/monitor_silu_500ep

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== atrophy (500 ep) ==="
$PYTHON scripts/train.py \
    --mode atrophy \
    --skip-ae \
    --ae-checkpoint $AE_CKPT \
    --diff-epochs 500 \
    --patience 600 \
    --val-every 1 \
    --monitor-every 25 \
    --monitor-dir results/figures/monitor_silu_500ep \
    --checkpoint-dir results/checkpoints/silu/atrophy_500ep \
    --use-mask
echo "atrophy done at: $(date)"
echo ""

echo "=== ptau217 (500 ep) ==="
$PYTHON scripts/train.py \
    --mode ptau217 \
    --skip-ae \
    --ae-checkpoint $AE_CKPT \
    --diff-epochs 500 \
    --patience 600 \
    --val-every 1 \
    --monitor-every 25 \
    --monitor-dir results/figures/monitor_silu_500ep \
    --checkpoint-dir results/checkpoints/silu/ptau217_500ep \
    --use-mask
echo "ptau217 done at: $(date)"
echo ""

echo "=== combined (500 ep) ==="
$PYTHON scripts/train.py \
    --mode combined \
    --skip-ae \
    --ae-checkpoint $AE_CKPT \
    --diff-epochs 500 \
    --patience 600 \
    --val-every 1 \
    --monitor-every 25 \
    --monitor-dir results/figures/monitor_silu_500ep \
    --checkpoint-dir results/checkpoints/silu/combined_500ep \
    --use-mask
echo "combined done at: $(date)"
echo ""

echo "All 500-epoch runs complete. End time: $(date)"
