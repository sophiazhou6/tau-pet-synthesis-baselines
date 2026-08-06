#!/bin/bash
# Leaderboard across every CFG-matrix cell (all w), on SET1/2/3 + WITHIN + PEAK.
cd "$(dirname "$0")/../.." || exit 1
PY=/usr/bin/python3.9
echo "=== FINAL-family cells (atrophy MLP / plasma-alone / combined) ==="
$PY scripts/peak_from_gens.py --dataset combined --mode combined_mlp --use-mentor-split \
    results/generated/cfg_matrix/cfgm_2_*/w* results/generated/cfg_matrix/cfgm_6_*/w* \
    results/generated/cfg_matrix/cfgm_8_*/w* results/generated/cfg_matrix/cfgm_9_*/w* 2>/dev/null
echo "=== SPATIAL-family cells (atrophy map +/- plasma, tissue) ==="
$PY scripts/peak_from_gens.py --dataset spatial --mode atrophy --cond-mode ptau217 --use-mentor-split \
    results/generated/cfg_matrix/cfgm_1_*/w* results/generated/cfg_matrix/cfgm_4_*/w* \
    results/generated/cfg_matrix/cfgm_5_*/w* results/generated/cfg_matrix/cfgm_7_*/w* 2>/dev/null
