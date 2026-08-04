#!/bin/bash
#SBATCH --job-name=train_ptau217_mlp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/ptau217_mlp_%j.out
#SBATCH --error=results/logs/ptau217_mlp_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
mkdir -p results/checkpoints/silu/ptau217_mlp results/logs

echo "=== train_ptau217_mlp: start $(date) ==="

$PYTHON scripts/train.py \
    --mode ptau217_mlp \
    --skip-ae \
    --ae-checkpoint results/checkpoints/silu/atrophy/diff_atrophy.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --checkpoint-dir results/checkpoints/silu/ptau217_mlp \
    --use-mask

echo "=== train_ptau217_mlp: done $(date) ==="
