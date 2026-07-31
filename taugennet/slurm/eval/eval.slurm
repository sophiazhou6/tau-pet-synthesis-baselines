#!/bin/bash
#SBATCH --job-name=taugennet_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=results/logs/eval_%j.out
#SBATCH --error=results/logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/scratch/network/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet

echo "=== Evaluation: atrophy mode ==="
$PYTHON scripts/evaluate.py --mode atrophy
echo ""

echo "=== Evaluation: ptau217 mode ==="
$PYTHON scripts/evaluate.py --mode ptau217
echo ""

echo "Done. End time: $(date)"
