#!/bin/bash
#SBATCH --job-name=eval_large
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_large_%j.out
#SBATCH --error=results/logs/eval_large_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
mkdir -p results/logs results/figures/large results/records/large

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Atrophy ───────────────────────────────────────────────────────────────────
echo "=== Evaluating large/atrophy ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy \
    --checkpoint-dir results/checkpoints/large_arch \
    --figures-dir large \
    --records-dir results/records/large \
    --unet-channels 256,512,768 \
    --n-transformer 3 \
    --use-best \
    --save-generated \
    --generated-dir results/generated/large/atrophy \
    --skip-ablation
echo "Atrophy done at: $(date)"

# ── Combined ──────────────────────────────────────────────────────────────────
echo "=== Evaluating large/combined ==="
$PYTHON scripts/evaluate_final.py \
    --mode combined \
    --checkpoint-dir results/checkpoints/large_arch \
    --figures-dir large \
    --records-dir results/records/large \
    --unet-channels 256,512,768 \
    --n-transformer 3 \
    --use-best \
    --save-generated \
    --generated-dir results/generated/large/combined \
    --skip-ablation
echo "Combined done at: $(date)"

echo "All eval complete. End: $(date)"
