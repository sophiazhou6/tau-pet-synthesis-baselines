#!/bin/bash
#SBATCH --job-name=eval_small
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_small_%j.out
#SBATCH --error=results/logs/eval_small_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet
mkdir -p results/logs results/figures/small results/records/small

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Atrophy ───────────────────────────────────────────────────────────────────
echo "=== Evaluating small/atrophy ==="
$PYTHON scripts/evaluate_final.py \
    --mode atrophy \
    --checkpoint-dir results/checkpoints/small_arch \
    --figures-dir small \
    --records-dir results/records/small \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --use-best \
    --save-generated \
    --generated-dir results/generated/small/atrophy \
    --skip-ablation
echo "Atrophy done at: $(date)"

# ── Combined ──────────────────────────────────────────────────────────────────
echo "=== Evaluating small/combined ==="
$PYTHON scripts/evaluate_final.py \
    --mode combined \
    --checkpoint-dir results/checkpoints/small_arch \
    --figures-dir small \
    --records-dir results/records/small \
    --unet-channels 128,256,512 \
    --n-transformer 1 \
    --use-best \
    --save-generated \
    --generated-dir results/generated/small/combined \
    --skip-ablation
echo "Combined done at: $(date)"

echo "All eval complete. End: $(date)"
