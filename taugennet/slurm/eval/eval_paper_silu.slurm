#!/bin/bash
#SBATCH --job-name=eval_paper_silu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=results/logs/eval_paper_silu_%j.out
#SBATCH --error=results/logs/eval_paper_silu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3

cd /scratch/network/sz3962/taugennet

mkdir -p results/figures/silu/atrophy/paper_eval \
         results/figures/silu/ptau217/paper_eval \
         results/figures/silu/combined/paper_eval \
         results/records

echo "=== Paper eval: atrophy (silu) — $(date) ==="
$PYTHON scripts/evaluation_paper.py \
    --mode atrophy \
    --checkpoint-dir results/checkpoints/silu/atrophy \
    --figures-dir silu/atrophy \
    --records-dir results/records \
    --arch silu \
    --use-mask
echo "Atrophy done at: $(date)"

echo "=== Paper eval: ptau217 (silu) — $(date) ==="
$PYTHON scripts/evaluation_paper.py \
    --mode ptau217 \
    --checkpoint-dir results/checkpoints/silu/ptau217 \
    --figures-dir silu/ptau217 \
    --records-dir results/records \
    --arch silu \
    --use-mask
echo "ptau217 done at: $(date)"

echo "=== Paper eval: combined (silu) — $(date) ==="
$PYTHON scripts/evaluation_paper.py \
    --mode combined \
    --checkpoint-dir results/checkpoints/silu/combined \
    --figures-dir silu/combined \
    --records-dir results/records \
    --arch silu \
    --use-mask
echo "Combined done at: $(date)"

echo "All SiLU paper evaluations complete. End time: $(date)"
