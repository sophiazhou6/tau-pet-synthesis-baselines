#!/bin/bash
#SBATCH --job-name=eval_silu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_silu_%j.out
#SBATCH --error=results/logs/eval_silu_%j.err

PYTHON=/usr/bin/python3.9

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

mkdir -p results/figures/silu/atrophy \
         results/figures/silu/ptau217 \
         results/figures/silu/combined

echo "=== Evaluating atrophy (silu, masked) ==="
$PYTHON scripts/evaluate.py \
    --mode atrophy \
    --checkpoint-dir results/checkpoints/silu/atrophy \
    --figures-dir silu/atrophy \
    --use-mask \
    --arch silu
echo "Atrophy done at: $(date)"

echo "=== Evaluating ptau217 (silu, masked) ==="
$PYTHON scripts/evaluate.py \
    --mode ptau217 \
    --checkpoint-dir results/checkpoints/silu/ptau217 \
    --figures-dir silu/ptau217 \
    --use-mask \
    --arch silu
echo "ptau217 done at: $(date)"

echo "=== Evaluating combined (silu, masked) ==="
$PYTHON scripts/evaluate.py \
    --mode combined \
    --checkpoint-dir results/checkpoints/silu/combined \
    --figures-dir silu/combined \
    --use-mask \
    --arch silu
echo "Combined done at: $(date)"

echo "All SiLU evaluations complete. End time: $(date)"
