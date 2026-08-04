#!/bin/bash
#SBATCH --job-name=taugennet_figures
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/paper_figures_%j.out
#SBATCH --error=results/logs/paper_figures_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3

cd /scratch/network/sz3962/taugennet

mkdir -p results/figures/combined/paper \
         results/figures/ptau217/paper \
         results/figures/atrophy/paper

echo "=== Generating paper figures (all modes) ==="
$PYTHON scripts/paper_figures.py --mode all
echo "Done at: $(date)"
