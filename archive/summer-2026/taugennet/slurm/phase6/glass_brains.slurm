#!/bin/bash
#SBATCH --job-name=glass_brains
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/glass_brains_%j.out
set -x
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
export TAUGENNET_ROOT=$HOME/taugennet
cd $HOME/taugennet
PY=/usr/bin/python3.9

echo "===== 1.3 : 917734 SUVR (c13) fold0 w1 ====="
export TAUGENNET_SPATIAL_MAP=suvr
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --fold 0 --n-folds 5 --use-cached --generated-dir results/generated/suvr_spatial/fold_0/w1 --use-mask --use-mentor-split --skip-ablation --glass-brain --figures-dir glass/suvr_c13_fold0_w1 --records-dir results/records/_glass_tmp/suvr_c13_fold0_w1

echo "===== 1.1 : final c13_e520 (atrophy map) ====="
unset TAUGENNET_SPATIAL_MAP
GEN=$(dirname "$(find results/generated/final -path '*c13_e520*' -name 'subject_000.npy' 2>/dev/null | head -1)")
echo "c13_e520 gens = $GEN"
if [ -n "$GEN" ] && [ "$GEN" != "." ]; then
  $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --fold 0 --n-folds 5 --use-cached --generated-dir "$GEN" --use-mask --use-mentor-split --skip-ablation --glass-brain --figures-dir glass/final_c13_e520 --records-dir results/records/_glass_tmp/final_c13_e520
else
  echo "!! could not locate c13_e520 gens under results/generated/final"
fi
echo "[glass_brains] DONE"
find results/figures/glass -type d -name glass_brain
