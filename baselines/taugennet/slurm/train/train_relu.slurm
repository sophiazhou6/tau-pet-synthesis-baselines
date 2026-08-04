#!/bin/bash
#SBATCH --job-name=taugennet_relu
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/taugennet_relu_%j.out
#SBATCH --error=results/logs/taugennet_relu_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Paper-aligned architecture run (v3):
#   LATENT_CH=3, UNet [256,512,768], 6 transformer blocks/level, GroupNorm 32
#   70/10/20 train/val/test split  (dataset_v2.py)
#   Three diffusion modes: atrophy, ptau217, combined

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3

cd /scratch/network/sz3962/taugennet

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p results/logs \
         results/checkpoints/atrophy_relu \
         results/checkpoints/ptau217_relu \
         results/checkpoints/combined_relu \
         results/figures/monitor_relu

# ── Info ──────────────────────────────────────────────────────────────────────
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
echo "Working dir: $(pwd)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Stage 1: AE checkpoint already trained (job 3259943) ─────────────────────
echo "=== Stage 1: Skipping AE pretraining (checkpoint already exists) ==="
echo "Using: results/checkpoints/taugennet_checkpoint_relu.pt"
echo ""

# ── Stage 2: Diffusion — atrophy mode ────────────────────────────────────────
echo "=== Stage 2: Diffusion training (atrophy, paper arch) ==="
$PYTHON scripts/train_relu.py \
    --mode atrophy \
    --skip-ae \
    --ae-checkpoint results/checkpoints/taugennet_checkpoint_relu.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --monitor-dir results/figures/monitor_relu \
    --checkpoint-dir results/checkpoints/atrophy_relu
echo "Stage 2 done at: $(date)"
echo ""

# ── Stage 3: Diffusion — ptau217 mode ────────────────────────────────────────
echo "=== Stage 3: Diffusion training (ptau217, paper arch) ==="
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
echo ""

# ── Stage 4: Diffusion — combined (atrophy + ptau217) mode ───────────────────
echo "=== Stage 4: Diffusion training (combined, paper arch) ==="
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
echo ""

echo "All stages complete. End time: $(date)"
