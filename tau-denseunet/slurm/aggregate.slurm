#!/bin/bash
#SBATCH --job-name=denseunet_agg
#SBATCH --partition=cpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/agg_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/agg_%j.err

# Post-CV aggregation (CPU, no GPU). Queue AFTER the fold array with:
#   sbatch --dependency=afterok:<ARRAY_JOBID> slurm/aggregate.slurm atrophy dk86
#   sbatch --dependency=afterok:<ARRAY_JOBID> slurm/aggregate.slurm atrophy wholebrain
# Produces: CV mean±std tables, the 4 SUVR figures + glass brain (dk86) from the
# fold-ensemble, and CV per-region error-bar bars.
set -euo pipefail

MODE="${1:-atrophy}"
MASK="${2:-dk86}"
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
export CUDA_VISIBLE_DEVICES=""

cd "$REPO"
source $HOME/taugennet/scripts/wait_for_fac.sh
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:/home/remote/adebnath1/tau-denseunet-baseline:$HOME/taugennet:${PYTHONPATH:-}"
echo "Aggregation: mode=$MODE mask=$MASK   $(date)"
$PYTHON scripts/aggregate_cv.py --mode "$MODE" --mask-mode "$MASK"
echo "Done: $(date)"
