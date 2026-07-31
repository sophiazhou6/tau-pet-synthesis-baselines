#!/bin/bash
#SBATCH --job-name=eval_paper_relu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=results/logs/eval_paper_relu_%j.out
#SBATCH --error=results/logs/eval_paper_relu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3

cd /scratch/network/sz3962/taugennet

mkdir -p results/figures/relu/atrophy/paper_eval \
         results/figures/relu/ptau217/paper_eval \
         results/figures/relu/combined/paper_eval \
         results/records

echo "=== Paper eval: atrophy (relu) — $(date) ==="
$PYTHON scripts/evaluation_paper.py \
    --mode atrophy \
    --checkpoint-dir results/checkpoints/relu/atrophy \
    --figures-dir relu/atrophy \
    --records-dir results/records \
    --arch relu \
    --use-mask
echo "Atrophy done at: $(date)"

echo "=== Paper eval: ptau217 (relu) — $(date) ==="
$PYTHON scripts/evaluation_paper.py \
    --mode ptau217 \
    --checkpoint-dir results/checkpoints/relu/ptau217 \
    --figures-dir relu/ptau217 \
    --records-dir results/records \
    --arch relu \
    --use-mask
echo "ptau217 done at: $(date)"

echo "=== Paper eval: combined (relu) — $(date) ==="
$PYTHON scripts/evaluation_paper.py \
    --mode combined \
    --checkpoint-dir results/checkpoints/relu/combined \
    --figures-dir relu/combined \
    --records-dir results/records \
    --arch relu \
    --use-mask
echo "Combined done at: $(date)"

echo "All ReLU paper evaluations complete. End time: $(date)"
