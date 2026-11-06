#!/bin/bash
#SBATCH --job-name=eval_large
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_large_%j.out
#SBATCH --error=results/logs/eval_large_%j.err

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
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
