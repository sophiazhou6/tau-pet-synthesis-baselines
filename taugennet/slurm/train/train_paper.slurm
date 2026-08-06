#!/bin/bash
#SBATCH --job-name=taugennet_paper
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/taugennet_paper_%j.out
#SBATCH --error=results/logs/taugennet_paper_%j.err

# Paper-aligned architecture run (v3):
#   LATENT_CH=3, UNet [256,512,768], 6 transformer blocks/level, GroupNorm 32
#   70/10/20 train/val/test split  (dataset_v2.py)
#   Three diffusion modes: atrophy, ptau217, combined

PYTHON=/usr/bin/python3.9

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p results/logs \
         results/checkpoints/atrophy_paper \
         results/checkpoints/ptau217_paper \
         results/checkpoints/combined_paper \
         results/figures/monitor_paper

# ── Info ──────────────────────────────────────────────────────────────────────
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
echo "Working dir: $(pwd)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Stage 1: Shared AE pretraining ───────────────────────────────────────────
# Must retrain from scratch — new architecture (LATENT_CH=3) is incompatible
# with existing checkpoints.  AE is shared across all three diffusion modes.
echo "=== Stage 1: AE pretraining ==="
$PYTHON scripts/train.py \
    --mode atrophy \
    --ae-epochs 400 \
    --diff-epochs 0 \
    --ae-checkpoint results/checkpoints/taugennet_checkpoint_paper.pt
echo "Stage 1 done at: $(date)"
echo ""

# ── Stage 2: Diffusion — atrophy mode ────────────────────────────────────────
echo "=== Stage 2: Diffusion training (atrophy, paper arch) ==="
$PYTHON scripts/train.py \
    --mode atrophy \
    --skip-ae \
    --ae-checkpoint results/checkpoints/taugennet_checkpoint_paper.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --monitor-dir results/figures/monitor_paper \
    --checkpoint-dir results/checkpoints/atrophy_paper
echo "Stage 2 done at: $(date)"
echo ""

# ── Stage 3: Diffusion — ptau217 mode ────────────────────────────────────────
echo "=== Stage 3: Diffusion training (ptau217, paper arch) ==="
$PYTHON scripts/train.py \
    --mode ptau217 \
    --skip-ae \
    --ae-checkpoint results/checkpoints/taugennet_checkpoint_paper.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --monitor-dir results/figures/monitor_paper \
    --checkpoint-dir results/checkpoints/ptau217_paper
echo "Stage 3 done at: $(date)"
echo ""

# ── Stage 4: Diffusion — combined (atrophy + ptau217) mode ───────────────────
echo "=== Stage 4: Diffusion training (combined, paper arch) ==="
$PYTHON scripts/train.py \
    --mode combined \
    --skip-ae \
    --ae-checkpoint results/checkpoints/taugennet_checkpoint_paper.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --monitor-dir results/figures/monitor_paper \
    --checkpoint-dir results/checkpoints/combined_paper
echo "Stage 4 done at: $(date)"
echo ""

echo "All stages complete. End time: $(date)"
