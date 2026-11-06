#!/bin/bash
#SBATCH --job-name=taugennet_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=results/logs/eval_%j.out
#SBATCH --error=results/logs/eval_%j.err

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

echo "=== Evaluation: atrophy mode ==="
$PYTHON scripts/evaluate.py --mode atrophy
echo ""

echo "=== Evaluation: ptau217 mode ==="
$PYTHON scripts/evaluate.py --mode ptau217
echo ""

echo "Done. End time: $(date)"
