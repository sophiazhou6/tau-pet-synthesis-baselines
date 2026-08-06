#!/bin/bash
#SBATCH --job-name=roi_table_silu
#SBATCH --output=results/logs/roi_table_silu_%j.out
#SBATCH --error=results/logs/roi_table_silu_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-06:00:00

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

echo "Starting ROI table computation: $(date)"
$PYTHON scripts/roi_table.py --volumes-dir results/generated/silu
echo "Done: $(date)"
