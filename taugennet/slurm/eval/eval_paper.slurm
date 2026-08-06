#!/bin/bash
#SBATCH --job-name=taugennet_eval_paper
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_paper_%j.out
#SBATCH --error=results/logs/eval_paper_%j.err

PYTHON=/usr/bin/python3.9

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

mkdir -p results/figures/atrophy/paper \
         results/figures/ptau217/paper \
         results/figures/combined/paper

echo "=== Evaluating atrophy (paper) ==="
$PYTHON scripts/evaluate.py \
    --mode atrophy \
    --checkpoint-dir results/checkpoints/atrophy_paper \
    --figures-dir atrophy/paper
echo "Atrophy done at: $(date)"

echo "=== Evaluating ptau217 (paper) ==="
$PYTHON scripts/evaluate.py \
    --mode ptau217 \
    --checkpoint-dir results/checkpoints/ptau217_paper \
    --figures-dir ptau217/paper
echo "ptau217 done at: $(date)"

echo "=== Evaluating combined (paper) ==="
$PYTHON scripts/evaluate.py \
    --mode combined \
    --checkpoint-dir results/checkpoints/combined_paper \
    --figures-dir combined/paper
echo "Combined done at: $(date)"

echo "All evaluations complete. End time: $(date)"
