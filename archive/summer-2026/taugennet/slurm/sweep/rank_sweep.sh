#!/bin/bash
# Rank every finished sweep cell on all five metrics (reads cached .npy directly).
cd "$(dirname "$0")/../.." || exit 1
PY=/home/sz3962/.conda/envs/taugennet/bin/python3
COND_MODE=${1:-combined}
DS=$([ "$COND_MODE" = "atrophy" ] && echo final || echo combined)
$PY scripts/peak_from_gens.py --dataset $DS --mode $COND_MODE --use-mentor-split \
    results/generated/sweep/*/fold_0
