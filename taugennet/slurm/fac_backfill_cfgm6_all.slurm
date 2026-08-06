#!/bin/bash
#SBATCH --job-name=fac_backfill6
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:30:00
#SBATCH --output=results/logs/fac_backfill6_%j.out
#SBATCH --error=results/logs/fac_backfill6_%j.err
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

VARIANT_DIR=$(ls -d results/generated/cfg_matrix/cfgm_6_* 2>/dev/null | head -1)
VARIANT=$(basename "$VARIANT_DIR")
echo "cfgm_6 variant dir: $VARIANT_DIR"

for WDIR in "$VARIANT_DIR"/w*; do
  W=$(basename "$WDIR" | sed 's/^w//')
  echo "=== $VARIANT w$W ==="
  $PY scripts/evaluate_final.py --mode combined --dataset combined --cond-mode ptau217 \
      --use-mentor-split --use-mask --use-cached --glass-brain \
      --unet-channels 128,256,512 --n-transformer 1 \
      --generated-dir "results/generated/cfg_matrix/${VARIANT}/w${W}" \
      --figures-dir "cfg_matrix/${VARIANT}/w${W}" \
      --records-dir "results/records/cfg_matrix/${VARIANT}/fold_0/w${W}"
done

bash scripts/publish_to_fac.sh

FAC_DB=/mnt/fac/CX500024_DS2/ADNI_Data_INR/Diffusion_Baseline
echo "cfgm_6: $(find "$FAC_DB/results/figures/cfg_matrix" -path '*cfgm_6_*' -name '*.png' 2>/dev/null | wc -l) PNGs"
