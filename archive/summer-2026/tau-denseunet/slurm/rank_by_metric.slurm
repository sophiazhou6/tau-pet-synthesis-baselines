#!/bin/bash
#SBATCH --job-name=denseunet_rank
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/rank_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/rank_%j.err
set -euo pipefail
PYTHON=/usr/bin/python3.9
REPO=/home/remote/adebnath1/tau-denseunet-baseline
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
cd "$REPO"
source "$HOME/taugennet/scripts/wait_for_fac.sh"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$REPO:$HOME/taugennet:${PYTHONPATH:-}"

RANK_BY="${1:-mae}"   # <-- the lever: pearson | ssim | mae
echo "Ranking by: $RANK_BY"

for FOLD in 0 1 2 3 4; do
  echo "── item 1 (grid CV) fold $FOLD — rank_by=$RANK_BY ──"
  $PYTHON scripts/evaluate.py --mode atrophy --fold "$FOLD" \
      --generated-dir results/generated/atrophy_tuned_lr1e-4_wd0/fold_$FOLD \
      --rank-by "$RANK_BY"

  echo "── item 4a (painted SUVR) fold $FOLD — rank_by=$RANK_BY ──"
  $PYTHON scripts/evaluate.py --mode atrophy --fold "$FOLD" \
      --generated-dir results/generated/atrophy_final_suvr/fold_$FOLD \
      --out-tag suvr --rank-by "$RANK_BY"

  echo "── item 4b (FiLM) fold $FOLD — rank_by=$RANK_BY ──"
  $PYTHON scripts/evaluate.py --mode atrophy --fold "$FOLD" \
      --generated-dir results/generated/atrophy_final_film/fold_$FOLD \
      --out-tag film --rank-by "$RANK_BY"
done

echo "── Publishing to FAC ──"
bash "$HOME/taugennet/scripts/publish_to_fac.sh"
