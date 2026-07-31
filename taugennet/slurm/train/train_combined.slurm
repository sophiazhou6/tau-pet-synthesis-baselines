#!/bin/bash
#SBATCH --job-name=taugennet_combined
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/taugennet_combined_%j.out
#SBATCH --error=results/logs/taugennet_combined_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/scratch/network/sz3962/.conda/envs/taugennet/bin/python3

cd /scratch/network/sz3962/taugennet

mkdir -p results/logs results/checkpoints/combined \
         results/figures results/figures/monitor_combined

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
echo "Working dir: $(pwd)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Combined conditioning: atrophy z-scores (86-dim MLP) + ptau217 scalar (CLIP).
# Dataset: triple-matched subjects (MRI + PET + atrophy + ptau217), 80/10/10 split.
# Checkpoints saved separately so v1/v2 results are untouched.

echo "=== Combined conditioning: atrophy + ptau217 ==="
$PYTHON scripts/train.py \
    --mode combined \
    --skip-ae \
    --ae-checkpoint results/checkpoints/atrophy/diff_atrophy_old.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --checkpoint-dir results/checkpoints/combined
echo "Combined stage done at: $(date)"

echo "All stages complete. End time: $(date)"
