#!/bin/bash
#SBATCH --job-name=taugennet_v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/taugennet_v2_%j.out
#SBATCH --error=results/logs/taugennet_v2_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
PYTHON=/usr/bin/python3.9

cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p results/logs results/checkpoints/atrophy_v2 results/checkpoints/ptau217_v2 \
         results/figures results/figures/monitor_v2

# ── Info ──────────────────────────────────────────────────────────────────────
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
echo "Working dir: $(pwd)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Uses 80/10/10 split (dataset_v2.py) and separate checkpoint dirs so the
# original v1 checkpoints are untouched.

# ── Stage 2: Diffusion training — atrophy mode ───────────────────────────────
echo "=== Stage 2: Diffusion training (atrophy mode, v2 split) ==="
$PYTHON scripts/train.py \
    --mode atrophy \
    --skip-ae \
    --ae-checkpoint results/checkpoints/atrophy/diff_atrophy_old.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --checkpoint-dir results/checkpoints/atrophy_v2
echo "Stage 2 done at: $(date)"
echo ""

# ── Stage 3: Diffusion training — ptau217 mode ───────────────────────────────
echo "=== Stage 3: Diffusion training (ptau217 mode, v2 split) ==="
$PYTHON scripts/train.py \
    --mode ptau217 \
    --skip-ae \
    --ae-checkpoint results/checkpoints/atrophy/diff_atrophy_old.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --checkpoint-dir results/checkpoints/ptau217_v2
echo "Stage 3 done at: $(date)"
echo ""

echo "All stages complete. End time: $(date)"
