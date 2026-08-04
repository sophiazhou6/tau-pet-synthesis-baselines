#!/bin/bash
#SBATCH --job-name=taugen_s4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=results/logs/taugennet_relu_stage4_%j.out
#SBATCH --error=results/logs/taugennet_relu_stage4_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet

mkdir -p results/logs results/checkpoints/combined_relu results/figures/monitor_relu

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== Stage 4: Diffusion training (combined) ==="
$PYTHON scripts/train_relu.py \
    --mode combined \
    --skip-ae \
    --ae-checkpoint results/checkpoints/taugennet_checkpoint_relu.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --monitor-dir results/figures/monitor_relu \
    --checkpoint-dir results/checkpoints/combined_relu

echo "Stage 4 done at: $(date)"
