#!/bin/bash
#SBATCH --job-name=taugenne_relu_ae
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/taugennet_relu_ae_%j.out
#SBATCH --error=results/logs/taugennet_relu_ae_%j.err

PYTHON=/usr/bin/python3.9

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

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
