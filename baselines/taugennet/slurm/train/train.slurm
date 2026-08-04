#!/bin/bash
#SBATCH --job-name=taugennet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/taugennet_%j.out
#SBATCH --error=results/logs/taugennet_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# ── Environment ───────────────────────────────────────────────────────────────
PYTHON=/scratch/network/sz3962/.conda/envs/taugennet/bin/python3

cd /scratch/network/sz3962/taugennet

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p results/logs results/checkpoints/atrophy results/checkpoints/ptau217 \
         results/figures results/figures/monitor

# ── Info ──────────────────────────────────────────────────────────────────────
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
echo "Working dir: $(pwd)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# AE is loaded from the previous checkpoint — no AE retraining needed.
# The old diffusion checkpoints are renamed before each stage so training
# starts fresh (not resumed from the broken pre-fix run).

# ── Stage 2: Diffusion training — atrophy mode ───────────────────────────────
mv results/checkpoints/atrophy/diff_atrophy.pt \
   results/checkpoints/atrophy/diff_atrophy_old.pt 2>/dev/null || true
echo "=== Stage 2: Diffusion training (atrophy mode) ==="
$PYTHON scripts/train.py \
    --mode atrophy \
    --skip-ae \
    --ae-checkpoint results/checkpoints/atrophy/diff_atrophy_old.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --checkpoint-dir results/checkpoints/atrophy
echo "Stage 2 done at: $(date)"
echo ""

# ── Stage 3: Diffusion training — ptau217 mode ───────────────────────────────
mv results/checkpoints/ptau217/diff_ptau217.pt \
   results/checkpoints/ptau217/diff_ptau217_old.pt 2>/dev/null || true
echo "=== Stage 3: Diffusion training (ptau217 mode) ==="
$PYTHON scripts/train.py \
    --mode ptau217 \
    --skip-ae \
    --ae-checkpoint results/checkpoints/atrophy/diff_atrophy_old.pt \
    --diff-epochs 2000 \
    --patience 50 \
    --val-every 1 \
    --monitor-every 25 \
    --checkpoint-dir results/checkpoints/ptau217
echo "Stage 3 done at: $(date)"
echo ""

echo "All stages complete. End time: $(date)"
