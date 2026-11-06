#!/bin/bash
# launch_lr_cv.sh — submit LR grid search with 5-fold CV.
#
# Workflow:
#   Step 1  (5-fold array): train AE per fold with kl_weight=1e-4, no diffusion
#   Step 2  (3 × 5-fold arrays): train diffusion per fold for each LR value,
#           reusing the Step 1 AE checkpoints.
#
# Usage:
#   bash slurm/grid_search/launch_lr_cv.sh
#   bash slurm/grid_search/launch_lr_cv.sh --qos gpu-test   # high-priority queue

set -euo pipefail

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet

QOS=""
if [[ "${1:-}" == "--qos" ]]; then
    QOS="--qos $2"
fi

mkdir -p results/logs slurm/grid_search results/grid_search

# ── Generate SLURM scripts ────────────────────────────────────────────────────
echo "Generating Step 1 (AE-only) and Step 2 (LR sweep) SLURM scripts..."
$PYTHON scripts/run_grid_search.py --step 1 --kl-values 1e-4 --ae-only
$PYTHON scripts/run_grid_search.py --step 2 --kl-weight 1e-4

# ── Submit Step 1: AE per fold (array 0-4) ────────────────────────────────────
echo ""
echo "Submitting Step 1: AE training, 5 folds..."
S1_OUT=$(sbatch $QOS slurm/grid_search/step1_kl1e-04.slurm)
S1_JOB=$(echo "$S1_OUT" | awk '{print $NF}')
echo "  Step 1 job ID: $S1_JOB"

# ── Submit Step 2: LR sweep, depends on all Step 1 tasks completing ───────────
echo ""
echo "Submitting Step 2: LR sweep (3 values × 5 folds), dependency on Step 1..."
DEP="--dependency=afterok:${S1_JOB}"

for LR in 5e-05 1e-04 2e-04; do
    S2_OUT=$(sbatch $QOS $DEP slurm/grid_search/step2_lr${LR}.slurm)
    S2_JOB=$(echo "$S2_OUT" | awk '{print $NF}')
    echo "  lr=${LR}  job ID: $S2_JOB"
done

echo ""
echo "All jobs queued. Monitor with: squeue -u sz3962"
echo ""
echo "After Step 2 completes, collect results with:"
echo "  $PYTHON scripts/run_grid_search.py --collect-step2 --kl-weight 1e-4"
