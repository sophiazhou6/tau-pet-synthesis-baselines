#!/bin/bash
#SBATCH --job-name=eval_small
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=results/logs/eval_small_%j.out
#SBATCH --error=results/logs/eval_small_%j.err

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
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
