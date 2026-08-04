#!/bin/bash
#SBATCH --job-name=roi_table_silu
#SBATCH --output=results/logs/roi_table_silu_%j.out
#SBATCH --error=results/logs/roi_table_silu_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet

echo "Starting ROI table computation: $(date)"
$PYTHON scripts/roi_table.py --volumes-dir results/generated/silu
echo "Done: $(date)"
