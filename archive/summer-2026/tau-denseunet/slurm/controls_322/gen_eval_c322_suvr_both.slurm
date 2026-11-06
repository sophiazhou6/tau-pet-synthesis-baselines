#!/bin/bash
#SBATCH --job-name=du_c322_suvr_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/du_c322_suvr_eval_%j.out
#SBATCH --error=/home/remote/adebnath1/tau-denseunet-baseline/results/logs/du_c322_suvr_eval_%j.err

REPO=/home/remote/adebnath1/tau-denseunet-baseline
cd "$REPO"
export TAUGENNET_ROOT=/home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:$REPO:$TAUGENNET_ROOT:$PYTHONPATH"
PY=/usr/bin/python3.9

run_one () {
  TAG=$1; FLAG=$2; CKDIR=$3
  CK="$REPO/results/checkpoints/atrophy/$CKDIR/denseunet_best.pt"
  if [ ! -f "$CK" ]; then echo "SKIP $TAG -- no checkpoint at $CK"; return; fi
  echo "=== $TAG generate — $(date) ==="
  $PY scripts/generate.py --mode atrophy --use-controls-322 --split test $FLAG \
      --checkpoint "$CK" --out-dir "$REPO/results/generated/$CKDIR"
  if [ $? -ne 0 ]; then echo "$TAG GENERATION FAILED"; return; fi
  echo "=== $TAG evaluate — $(date) ==="
  $PY scripts/evaluate.py --mode atrophy \
      --generated-dir "$REPO/results/generated/$CKDIR" \
      --out-tag "$CKDIR" --use-controls-322 \
      --records-dir "$REPO/results/records/denseunet_baseline/$CKDIR"
  echo "$TAG done — $(date)"
}

run_one "painted-SUVR" "--use-suvr"      "controls322_suvr_lr1e-4_wd0"
run_one "FiLM-SUVR"    "--use-suvr-film" "controls322_suvrfilm_lr1e-4_wd0"
echo "=== ALL DONE $(date) ==="
