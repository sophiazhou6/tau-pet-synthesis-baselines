#!/bin/bash
#SBATCH --job-name=glass_mae5f
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/glass_mae5f_%j.out
set -x
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
export TAUGENNET_ROOT=$HOME/taugennet
cd $HOME/taugennet
PY=/usr/bin/python3.9
FAC=/mnt/fac/CX500024_DS2/ADNI_Data_INR/Diffusion_Baseline
MAN=results/records/phase6_manifest_v4.csv
MF=results/mentor_figures

echo "===== 1.3 SUVR — ALL 5 FOLDS, ranked by MAE ====="
export TAUGENNET_SPATIAL_MAP=suvr
for f in 0 1 2 3 4; do
  $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --fold $f --n-folds 5 \
    --use-cached --generated-dir results/generated/suvr_spatial/fold_$f/w1 --use-mask --use-mentor-split \
    --skip-ablation --glass-brain --rank-by mae \
    --figures-dir glass_mae/suvr_fold$f --records-dir results/records/_gm/suvr_fold$f
  mkdir -p $MF/C_suvr_5fold_917734/glass_brain_5fold/fold_$f
  cp -r results/figures/glass_mae/suvr_fold$f/glass_brain/mae $MF/C_suvr_5fold_917734/glass_brain_5fold/fold_$f/ 2>/dev/null
done

echo "===== 1.1/#1 final c13_e520 — ranked by MAE ====="
unset TAUGENNET_SPATIAL_MAP
GEN=$(dirname "$(find results/generated/final -path '*c13_e520*' -name 'subject_000.npy' 2>/dev/null | head -1)")
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --fold 0 --n-folds 5 \
  --use-cached --generated-dir "$GEN" --use-mask --use-mentor-split --skip-ablation --glass-brain --rank-by mae \
  --figures-dir glass_mae/final_c13 --records-dir results/records/_gm/final_c13
cp -r results/figures/glass_mae/final_c13/glass_brain/mae $MF/B_final_noctrl_c13/glass_brain/ 2>/dev/null

echo "===== 1.2 controls — ranked by MAE (if gens exist) ====="
G12=results/generated/phase6_final/c13_ptau_e520/w1
if ls "$G12"/subject_*.npy >/dev/null 2>&1; then
  $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --phase6-manifest $MAN \
    --fold 0 --n-folds 5 --use-cached --generated-dir "$G12" --use-mask --skip-ablation --glass-brain --rank-by mae \
    --figures-dir glass_mae/controls_c13 --records-dir results/records/_gm/controls
  mkdir -p $MF/D_final_with_controls/glass_brain
  cp -r results/figures/glass_mae/controls_c13/glass_brain/mae $MF/D_final_with_controls/glass_brain/ 2>/dev/null
fi

echo "===== sync B / C / D (incl. cv_summary + 5-fold scatter + all glass brains) to FAC ====="
for D in B_final_noctrl_c13 C_suvr_5fold_917734 D_final_with_controls; do
  [ -d "$MF/$D" ] && rsync -a "$MF/$D/" "$FAC/results/mentor_figures/$D/"
done
echo "[glass_mae5f] DONE"
echo "MAE best/median/worst per run (verify direction — best should be LOW mae):"
grep -iE "ranking|Best|Median|Worst|MAE=" results/logs/glass_mae5f_*.out 2>/dev/null | tail -30
find $MF/C_suvr_5fold_917734 -name "*.png" | sort
