#!/bin/bash
#SBATCH --job-name=taugenne_relu_ae
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/taugennet_relu_ae_%j.out
#SBATCH --error=results/logs/taugennet_relu_ae_%j.err

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3

cd /scratch/network/sz3962/taugennet

mkdir -p results/logs results/checkpoints

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Stage 1: AE pretraining (400 epochs) ==="
$PYTHON scripts/train_relu.py \
    --mode atrophy \
    --ae-epochs 400 \
    --diff-epochs 0 \
    --ae-checkpoint results/checkpoints/taugennet_checkpoint_relu.pt

echo ""
echo "End time: $(date)"
ls -lh results/checkpoints/taugennet_checkpoint_relu.pt
