#!/bin/bash
#SBATCH --job-name=eval_ptau217_mlp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/eval_ptau217_mlp_%j.out
#SBATCH --error=results/logs/eval_ptau217_mlp_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
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
