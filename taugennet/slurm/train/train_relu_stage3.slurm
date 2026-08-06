#!/bin/bash
#SBATCH --job-name=taugen_s3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=results/logs/taugennet_relu_stage3_%j.out
#SBATCH --error=results/logs/taugennet_relu_stage3_%j.err

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

mkdir -p results/logs results/checkpoints/ptau217_relu results/figures/monitor_relu

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Stage 3: Diffusion training (ptau217) ==="
$PYTHON scripts/train_relu.py \
    --mode ptau217 \
    --skip-ae \
    --ae-checkpoint results/checkpoints/taugennet_checkpoint_relu.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --monitor-dir results/figures/monitor_relu \
    --checkpoint-dir results/checkpoints/ptau217_relu

echo "Stage 3 done at: $(date)"
