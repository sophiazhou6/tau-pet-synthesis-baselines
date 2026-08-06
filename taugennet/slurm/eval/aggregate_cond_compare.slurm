#!/bin/bash
#SBATCH --job-name=agg_cond_compare
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/agg_cond_compare_%j.out
#SBATCH --error=results/logs/agg_cond_compare_%j.err

# Generate (DDPM, 500 steps) for every cond_compare variant × fold on the shared fixed
# test set, then pool per-subject in-DK86 metrics into a variant comparison table.
# Reuses scripts/generate_spatial.py (generation) and evaluate_final.py (metrics).
# Cached subject_NNN.npy are reused across re-runs; pass --regenerate to force.

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
mkdir -p results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Add atrophy:atrophy below if/when a 5-fold atrophy-MLP baseline is trained under cond_compare/.
$PYTHON scripts/aggregate_cond_compare.py \
    --variants spatialmap_none:none spatialmap_ptau:ptau217 \
    --root results/checkpoints/cond_compare \
    --generated-root results/generated/cond_compare \
    --records-dir results/records/cond_compare \
    --n-folds 5 --n-steps 500

echo "Done: $(date)"
