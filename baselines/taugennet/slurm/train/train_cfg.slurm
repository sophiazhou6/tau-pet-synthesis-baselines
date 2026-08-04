#!/bin/bash
#SBATCH --job-name=taugennet_cfg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=results/logs/taugennet_cfg_%j.out
#SBATCH --error=results/logs/taugennet_cfg_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3

cd /scratch/network/sz3962/taugennet

mkdir -p results/logs results/checkpoints/cfg/atrophy results/checkpoints/cfg/combined \
         results/figures/cfg/atrophy results/figures/cfg/combined \
         results/figures/cfg/atrophy/monitor results/figures/cfg/combined/monitor

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
echo "Working dir: $(pwd)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

AE_CKPT=results/checkpoints/ae_exp_masked/baseline/ae_checkpoint_best.pt

# ── Combined mode ─────────────────────────────────────────────────────────────
echo "=== CFG diffusion training: combined ==="
$PYTHON scripts/train_cfg.py \
    --mode combined \
    --skip-ae \
    --ae-checkpoint $AE_CKPT \
    --diff-epochs 2000 \
    --patience 100 \
    --val-every 1 \
    --monitor-every 25 \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --cfg-prob 0.15 \
    --guidance-scale 3.0 \
    --checkpoint-dir results/checkpoints/cfg/combined \
    --figures-dir results/figures/cfg/combined \
    --monitor-dir results/figures/cfg/combined/monitor
echo "Combined done at: $(date)"
echo ""

echo "CFG combined training complete. End time: $(date)"
