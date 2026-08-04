#!/bin/bash
#SBATCH --job-name=denseunet_agg
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/agg_%j.out
#SBATCH --error=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/agg_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# Post-CV aggregation (CPU, no GPU). Queue AFTER the fold array with:
#   sbatch --dependency=afterok:<ARRAY_JOBID> slurm/aggregate.slurm atrophy dk86
#   sbatch --dependency=afterok:<ARRAY_JOBID> slurm/aggregate.slurm atrophy wholebrain
# Produces: CV mean±std tables, the 4 SUVR figures + glass brain (dk86) from the
# fold-ensemble, and CV per-region error-bar bars.
set -euo pipefail

MODE="${1:-atrophy}"
MASK="${2:-dk86}"
PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
REPO=/scratch/network/sz3962/tau-denseunet-baseline
export TAUGENNET_ROOT=/scratch/network/sz3962/taugennet
export CUDA_VISIBLE_DEVICES=""

cd "$REPO"
echo "Aggregation: mode=$MODE mask=$MASK   $(date)"
$PYTHON scripts/aggregate_cv.py --mode "$MODE" --mask-mode "$MASK"
echo "Done: $(date)"
