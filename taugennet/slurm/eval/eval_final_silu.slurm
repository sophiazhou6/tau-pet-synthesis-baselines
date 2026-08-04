#!/bin/bash
#SBATCH --job-name=eval_final_v3_silu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/eval_final_v3_silu_%j.out
#SBATCH --error=results/logs/eval_final_v3_silu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3

cd /scratch/network/sz3962/taugennet

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
    --generated-dir /scratch/network/sz3962/taugennet/results/generated/silu_v3/atrophy
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
    --generated-dir /scratch/network/sz3962/taugennet/results/generated/silu_v3/ptau217
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
    --generated-dir /scratch/network/sz3962/taugennet/results/generated/silu_v3/combined
echo "Combined done at: $(date)"

echo "All SiLU final evaluations complete. End time: $(date)"
