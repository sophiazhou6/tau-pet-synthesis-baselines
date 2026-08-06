#!/bin/bash
#SBATCH --job-name=glass_pearson
#SBATCH --time=00:50:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=results/logs/glass_pearson_%j.out
set -x
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
export TAUGENNET_ROOT=$HOME/taugennet
cd $HOME/taugennet
PY=/usr/bin/python3.9
FAC=/mnt/fac/CX500024_DS2/ADNI_Data_INR/Diffusion_Baseline
MAN=results/records/phase6_manifest_v4.csv
echo "===== 1.3  SUVR 917734 fold0 w1  (rank_by=pearson) ====="
export TAUGENNET_SPATIAL_MAP=suvr
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --fold 0 --n-folds 5 \
  --use-cached --generated-dir results/generated/suvr_spatial/fold_0/w1 --use-mask --use-mentor-split \
  --skip-ablation --glass-brain --rank-by pearson \
  --figures-dir glass_pearson/suvr --records-dir results/records/_gp/suvr
cp -r results/figures/glass_pearson/suvr/glass_brain/pearson results/mentor_figures/C_suvr_5fold_917734/glass_brain/ 2>/dev/null
echo "===== 1.1/#1  final c13_e520 atrophy  (rank_by=pearson) ====="
unset TAUGENNET_SPATIAL_MAP
GEN=$(dirname "$(find results/generated/final -path '*c13_e520*' -name 'subject_000.npy' 2>/dev/null | head -1)")
echo "gens = $GEN"
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --fold 0 --n-folds 5 \
  --use-cached --generated-dir "$GEN" --use-mask --use-mentor-split \
  --skip-ablation --glass-brain --rank-by pearson \
  --figures-dir glass_pearson/final_c13 --records-dir results/records/_gp/final_c13
cp -r results/figures/glass_pearson/final_c13/glass_brain/pearson results/mentor_figures/B_final_noctrl_c13/glass_brain/ 2>/dev/null
echo "===== 1.2  controls final (only if gens exist yet) ====="
G12=results/generated/phase6_final/c13_ptau_e520/w1
if ls "$G12"/subject_*.npy >/dev/null 2>&1; then
  $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --phase6-manifest $MAN \
    --fold 0 --n-folds 5 --use-cached --generated-dir "$G12" --use-mask \
    --skip-ablation --glass-brain --rank-by pearson \
    --figures-dir glass_pearson/controls_c13 --records-dir results/records/_gp/controls
  mkdir -p results/mentor_figures/D_final_with_controls/glass_brain
  cp -r results/figures/glass_pearson/controls_c13/glass_brain/pearson results/mentor_figures/D_final_with_controls/glass_brain/ 2>/dev/null
else
  echo ">> 1.2 gens not ready (991633 still training) — re-run this job after it finishes to add 1.2"
fi
echo "===== sync pearson glass brains to FAC ====="
for D in B_final_noctrl_c13 C_suvr_5fold_917734 D_final_with_controls; do
  [ -d "results/mentor_figures/$D/glass_brain/pearson" ] && rsync -a "results/mentor_figures/$D/glass_brain" "$FAC/results/mentor_figures/$D/"
done
echo "[glass_pearson] DONE"
find results/figures/glass_pearson -path "*pearson*" -name "glass_A.png" | sort
