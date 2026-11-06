#!/bin/bash
#SBATCH --job-name=denseunet_filmmed_eval
#SBATCH --partition=gpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/eval_filmmed_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/eval_filmmed_%j.err
set -euo pipefail
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
CKPT="$REPO/results/checkpoints/atrophy/final_film_median_lr1e-4_wd0/denseunet_best.pt"
GEN_DIR="$REPO/results/generated/atrophy_final_film_median"
cd "$REPO"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$REPO:$HOME/taugennet:${PYTHONPATH:-}"
mkdir -p results/logs "$GEN_DIR"

echo "-- 1/3 Generate (FiLM, median-epoch, test split) --"
$PYTHON scripts/generate.py --mode atrophy --mask-mode dk86 \
    --checkpoint "$CKPT" --out-dir "$GEN_DIR" --use-suvr-film --overwrite --mentor-split-dir /home/remote/adebnath1/taugennet/data/raw/legacy_admci_split

echo "-- 2/3 Scoring with TauGenNet evaluate_final --"
$PYTHON scripts/evaluate.py --mode atrophy --generated-dir "$GEN_DIR" --out-tag film_median

echo "-- 3/3 SUVR SET 1/2/3 --"
$PYTHON scripts/suvr_sets.py --mode atrophy --mask-mode dk86 --generated-dir "$GEN_DIR" --out-tag film_median --mentor-split-dir /home/remote/adebnath1/taugennet/data/raw/legacy_admci_split

echo "-- Publishing to FAC --"
bash "$HOME/taugennet/scripts/publish_to_fac.sh"
