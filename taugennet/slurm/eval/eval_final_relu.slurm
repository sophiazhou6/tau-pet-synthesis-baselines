#!/bin/bash
#SBATCH --job-name=eval_final_v3_relu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/eval_final_v3_relu_%j.out
#SBATCH --error=results/logs/eval_final_v3_relu_%j.err

PYTHON=/usr/bin/python3.9

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

mkdir -p results/figures/relu/atrophy/eval_final_v3 \
         results/figures/relu/ptau217/eval_final_v3 \
         results/figures/relu/combined/eval_final_v3 \
         results/records/relu \
         results/generated/relu_v3/atrophy \
         results/generated/relu_v3/ptau217 \
         results/generated/relu_v3/combined \
         results/logs

echo "=== eval_final_v3: atrophy (relu) — $(date) ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy \
    --checkpoint-dir results/checkpoints/relu/atrophy \
    --figures-dir relu/atrophy/eval_final_v3 \
    --records-dir results/records/relu \
    --arch relu \
    --use-mask \
    --n-steps 500 \
    --glass-brain \
    --save-generated \
    --generated-dir /home/remote/adebnath1/taugennet/results/generated/relu_v3/atrophy
echo "Atrophy done at: $(date)"

echo "=== eval_final_v3: ptau217 (relu) — $(date) ==="
$PYTHON scripts/evaluate_final.py \
    --mode ptau217 \
    --checkpoint-dir results/checkpoints/relu/ptau217 \
    --figures-dir relu/ptau217/eval_final_v3 \
    --records-dir results/records/relu \
    --arch relu \
    --use-mask \
    --n-steps 500 \
    --glass-brain \
    --save-generated \
    --generated-dir /home/remote/adebnath1/taugennet/results/generated/relu_v3/ptau217
echo "ptau217 done at: $(date)"

echo "=== eval_final_v3: combined (relu) — $(date) ==="
$PYTHON scripts/evaluate_final.py \
    --mode combined \
    --checkpoint-dir results/checkpoints/relu/combined \
    --figures-dir relu/combined/eval_final_v3 \
    --records-dir results/records/relu \
    --arch relu \
    --use-mask \
    --n-steps 500 \
    --glass-brain \
    --save-generated \
    --generated-dir /home/remote/adebnath1/taugennet/results/generated/relu_v3/combined
echo "Combined done at: $(date)"

echo "All ReLU final evaluations complete. End time: $(date)"
