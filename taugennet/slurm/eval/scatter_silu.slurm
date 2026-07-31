#!/bin/bash
#SBATCH --job-name=scatter_silu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=results/logs/scatter_silu_%j.out
#SBATCH --error=results/logs/scatter_silu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3

cd /scratch/network/sz3962/taugennet

mkdir -p results/figures/silu/atrophy/scatter \
         results/figures/silu/ptau217/scatter \
         results/figures/silu/combined/scatter \
         results/logs

echo "=== scatter: atrophy (silu) — $(date) ==="
$PYTHON scripts/scatter_plot.py --mode atrophy --arch silu --n-steps 500 --voxelwise
echo "Atrophy done at: $(date)"

echo "=== scatter: ptau217 (silu) — $(date) ==="
$PYTHON scripts/scatter_plot.py --mode ptau217 --arch silu --n-steps 500 --voxelwise
echo "ptau217 done at: $(date)"

echo "=== scatter: combined (silu) — $(date) ==="
$PYTHON scripts/scatter_plot.py --mode combined --arch silu --n-steps 500 --voxelwise
echo "Combined done at: $(date)"

echo "All scatter plots complete. End time: $(date)"
