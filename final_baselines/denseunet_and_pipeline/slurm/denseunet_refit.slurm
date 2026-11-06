#!/bin/bash
#SBATCH --job-name=du_refit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/du_refit_%j.out
#SBATCH --error=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/du_refit_%j.err
#SBATCH --mail-type=FAIL
#
# Dense-U-Net (published recipe) final model, after its 5-fold CV (submit with
# --dependency=afterok:<CV array>):
#   1. per-round median stopping point of the 5 folds          scripts/cv_round_schedule.py
#   2. CV column: each fold scored on its own validation subjects (CPUs, while the GPU trains)
#   3. refit on all 477 development subjects replaying that schedule, no validation set
#   4. held-out test: generate, score (GitHub pipeline), export NIfTIs, CV table with SDs
set -euo pipefail
PY=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/tau-denseunet-baseline
export TAUGENNET_ROOT=/scratch/network/sz3962/taugennet
CV_TAG="${CV_TAG:-denseunet_paper}"
TAG="${TAG:-denseunet_paper_refit}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

SCHED=$($PY scripts/cv_round_schedule.py --cohort cohort_v2 --tag "$CV_TAG" | tee /dev/stderr | awk '/^--round-schedule/{print $2}')
echo "── refit schedule: $SCHED ──"

( for k in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES= $PY scripts/score_baseline.py --manifest cohort_v2 --tag "$CV_TAG" --fold $k --split val \
        --use-controls-322 > "results/logs/valscore_${CV_TAG}_f$k.log" 2>&1 || echo "WARN: val scoring fold $k failed"
  done ) &
VALPID=$!

echo "── train ($(date)) ──"
$PY scripts/train_baseline.py --model denseunet_paper --tag "$TAG" --manifest cohort_v2 \
    --val-frac 0 --seed 102 --round-schedule "$SCHED"
echo "── generate + score held-out test ($(date)) ──"
$PY scripts/generate_baseline.py --model denseunet_paper --tag "$TAG" --manifest cohort_v2 --overwrite
$PY scripts/score_baseline.py --model denseunet_paper --tag "$TAG" --manifest cohort_v2 --use-controls-322
CUDA_VISIBLE_DEVICES= $PY scripts/export_predictions_nifti.py --cohort cohort_v2 --tag "$TAG" --split test

wait $VALPID
echo "── tables ($(date)) ──"
$PY scripts/cv_table.py --cohort cohort_v2 --cv-tag "$CV_TAG" --test-tag "$TAG"
$PY scripts/report_run.py --cohort cohort_v2 --tag "$TAG"
echo "DU REFIT OK ($(date))"
