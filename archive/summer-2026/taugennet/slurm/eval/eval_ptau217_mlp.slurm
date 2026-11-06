#!/bin/bash
#SBATCH --job-name=eval_ptau217_mlp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/eval_ptau217_mlp_%j.out
#SBATCH --error=results/logs/eval_ptau217_mlp_%j.err

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
mkdir -p results/figures/silu/ptau217_mlp/eval_final results/records/silu_ptau217_mlp results/logs

echo "=== eval_ptau217_mlp: start $(date) ==="

$PYTHON scripts/evaluate_final.py \
    --mode ptau217_mlp \
    --checkpoint-dir results/checkpoints/silu/ptau217_mlp \
    --figures-dir silu/ptau217_mlp/eval_final \
    --records-dir results/records/silu_ptau217_mlp \
    --arch silu \
    --use-mask \
    --n-steps 500 \
    --save-generated

echo "=== eval_ptau217_mlp: done $(date) ==="
