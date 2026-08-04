#!/bin/bash
#SBATCH --job-name=eval_relu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_relu_%j.out
#SBATCH --error=results/logs/eval_relu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3

cd /scratch/network/sz3962/taugennet

mkdir -p results/figures/relu/atrophy

echo "=== Evaluating atrophy (relu, masked) ==="
$PYTHON scripts/evaluate.py \
    --mode atrophy \
    --checkpoint-dir results/checkpoints/relu/atrophy \
    --figures-dir relu/atrophy \
    --use-mask \
    --arch relu
echo "Atrophy done at: $(date)"

echo "All ReLU evaluations complete. End time: $(date)"
