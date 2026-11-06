#!/bin/bash
#SBATCH --job-name=taugennet_ae
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=results/logs/taugennet_ae_%j.out
#SBATCH --error=results/logs/taugennet_ae_%j.err

# Retrain AE from scratch with post-activation ResBlock (Conv → GN → SiLU).
# Saves to results/checkpoints/taugennet_checkpoint_postact.pt
# Run this before train_large.slurm / train_small.slurm.

PYTHON=/usr/bin/python3.9
AE_CKPT=results/checkpoints/taugennet_checkpoint_postact.pt

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

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
