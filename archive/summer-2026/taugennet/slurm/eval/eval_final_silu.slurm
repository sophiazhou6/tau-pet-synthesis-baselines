#!/bin/bash
#SBATCH --job-name=eval_final_v3_silu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/eval_final_v3_silu_%j.out
#SBATCH --error=results/logs/eval_final_v3_silu_%j.err

PYTHON=/usr/bin/python3.9

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

mkdir -p results/figures/silu/atrophy/eval_final_v3 \
         results/figures/silu/ptau217/eval_final_v3 \
         results/figures/silu/combined/eval_final_v3 \
         results/records/silu \
         results/generated/silu_v3/atrophy \
         results/generated/silu_v3/ptau217 \
         results/generated/silu_v3/combined \
         results/logs

echo "=== eval_final_v3: atrophy (silu) — $(date) ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy \
    --checkpoint-dir results/checkpoints/silu/atrophy \
    --figures-dir silu/atrophy/eval_final_v3 \
    --records-dir results/records/silu \
    --arch silu \
    --use-mask \
    --n-steps 500 \
    --glass-brain \
    --save-generated \
    --generated-dir /home/remote/adebnath1/taugennet/results/generated/silu_v3/atrophy
echo "Atrophy done at: $(date)"

echo "=== eval_final_v3: ptau217 (silu) — $(date) ==="
$PYTHON scripts/evaluate_final.py \
    --mode ptau217 \
    --checkpoint-dir results/checkpoints/silu/ptau217 \
    --figures-dir silu/ptau217/eval_final_v3 \
    --records-dir results/records/silu \
    --arch silu \
    --use-mask \
    --n-steps 500 \
    --glass-brain \
    --save-generated \
    --generated-dir /home/remote/adebnath1/taugennet/results/generated/silu_v3/ptau217
echo "ptau217 done at: $(date)"

echo "=== eval_final_v3: combined (silu) — $(date) ==="
$PYTHON scripts/evaluate_final.py \
    --mode combined \
    --checkpoint-dir results/checkpoints/silu/combined \
    --figures-dir silu/combined/eval_final_v3 \
    --records-dir results/records/silu \
    --arch silu \
    --use-mask \
    --n-steps 500 \
    --glass-brain \
    --save-generated \
    --generated-dir /home/remote/adebnath1/taugennet/results/generated/silu_v3/combined
echo "Combined done at: $(date)"

echo "All SiLU final evaluations complete. End time: $(date)"
