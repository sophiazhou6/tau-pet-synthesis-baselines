#!/bin/bash
#SBATCH --job-name=eval_relu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_relu_%j.out
#SBATCH --error=results/logs/eval_relu_%j.err

PYTHON=/usr/bin/python3.9

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

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
