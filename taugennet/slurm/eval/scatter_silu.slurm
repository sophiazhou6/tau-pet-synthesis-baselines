#!/bin/bash
#SBATCH --job-name=scatter_silu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=results/logs/scatter_silu_%j.out
#SBATCH --error=results/logs/scatter_silu_%j.err

PYTHON=/usr/bin/python3.9

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

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
