#!/bin/bash
#SBATCH --job-name=tau_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=adroit-h11g2
#SBATCH --mem=48G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/smoke_%j.out
#SBATCH --error=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/smoke_%j.err

# GPU smoke test: (1) one training step + predict of every model on synthetic volumes
# (memory/time report), (2) a real end-to-end mini run (4 subjects, 1 epoch) of every model
# through train → generate → score on the given manifest, including one glass-brain render.
set -uo pipefail
MANIFEST="${1:-mentor235}"
PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
REPO=/scratch/network/sz3962/tau-denseunet-baseline
export TAUGENNET_ROOT="${TAUGENNET_ROOT:-/scratch/network/sz3962/taugennet}"
cd "$REPO"; mkdir -p results/logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== (1) synthetic smoke test ==="
$PYTHON scripts/smoke_test.py --batch-size 2
RC1=$?

echo "=== (2) end-to-end mini runs on $MANIFEST ==="
FAIL=""
for M in denseunet swinunetr eagan deagan ct2pet jointsyn_paired; do
  echo "--- $M ---"
  TAG="smoke_$M"
  $PYTHON scripts/train_baseline.py --model "$M" --tag "$TAG" --manifest "$MANIFEST" --fold 0 \
      --limit-subjects 4 --epochs 1 --epochs-share 1 --epochs-cls 1 --epochs-joint 1 --num-workers 0 --overwrite \
  && $PYTHON scripts/generate_baseline.py --model "$M" --tag "$TAG" --manifest "$MANIFEST" --fold 0 --limit-subjects 4 --overwrite \
  && $PYTHON scripts/score_baseline.py --model "$M" --tag "$TAG" --manifest "$MANIFEST" --fold 0 --limit-subjects 4 --pipeline local \
        $( [ "$M" = denseunet ] && echo "--glass-brain" || echo "--no-glass-brain" ) \
  || FAIL="$FAIL $M"
done
echo "=== synthetic smoke rc=$RC1 ; end-to-end failures:${FAIL:- none} ==="
[ $RC1 -eq 0 ] && [ -z "$FAIL" ]
