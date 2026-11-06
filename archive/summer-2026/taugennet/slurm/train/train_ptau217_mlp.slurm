#!/bin/bash
#SBATCH --job-name=train_ptau217_mlp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/ptau217_mlp_%j.out
#SBATCH --error=results/logs/ptau217_mlp_%j.err

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
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
