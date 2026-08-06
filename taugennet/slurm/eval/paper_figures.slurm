#!/bin/bash
#SBATCH --job-name=taugennet_figures
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/paper_figures_%j.out
#SBATCH --error=results/logs/paper_figures_%j.err

PYTHON=/usr/bin/python3.9

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

mkdir -p results/figures/combined/paper \
         results/figures/ptau217/paper \
         results/figures/atrophy/paper

echo "=== Generating paper figures (all modes) ==="
$PYTHON scripts/paper_figures.py --mode all
echo "Done at: $(date)"
