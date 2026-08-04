#!/bin/bash
#SBATCH --job-name=taugennet_ae
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=results/logs/taugennet_ae_%j.out
#SBATCH --error=results/logs/taugennet_ae_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Retrain AE from scratch with post-activation ResBlock (Conv → GN → SiLU).
# Saves to results/checkpoints/taugennet_checkpoint_postact.pt
# Run this before train_large.slurm / train_small.slurm.

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
AE_CKPT=results/checkpoints/taugennet_checkpoint_postact.pt

cd /scratch/network/sz3962/taugennet

mkdir -p results/logs results/checkpoints results/figures

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

$PYTHON scripts/train.py \
    --mode atrophy \
    --ae-checkpoint $AE_CKPT \
    --ae-epochs 100 \
    --patience 20 \
    --diff-epochs 0 \
    --figures-dir results/figures/ae

echo "AE training complete at: $(date)"
echo "Checkpoint saved to: $AE_CKPT"
