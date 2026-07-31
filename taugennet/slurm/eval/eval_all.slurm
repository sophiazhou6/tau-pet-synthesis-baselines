#!/bin/bash
#SBATCH --job-name=taugennet_eval_all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_all_%j.out
#SBATCH --error=results/logs/eval_all_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/scratch/network/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet

echo "=== atrophy / 72split ==="
$PYTHON scripts/evaluate.py --mode atrophy
echo ""

echo "=== ptau217 / 72split ==="
$PYTHON scripts/evaluate.py --mode ptau217
echo ""

echo "=== atrophy / 80split ==="
$PYTHON scripts/evaluate.py --mode atrophy --checkpoint-dir results/checkpoints/atrophy_v2
echo ""

echo "=== ptau217 / 80split ==="
$PYTHON scripts/evaluate.py --mode ptau217 --checkpoint-dir results/checkpoints/ptau217_v2
echo ""

echo "=== atrophy_ptau217 (combined) ==="
$PYTHON scripts/evaluate.py --mode combined
echo ""

echo "All evaluations done. End time: $(date)"
