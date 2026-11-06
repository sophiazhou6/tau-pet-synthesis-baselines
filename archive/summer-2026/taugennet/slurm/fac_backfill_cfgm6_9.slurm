#!/bin/bash
#SBATCH --job-name=fac_backfill
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:30:00
#SBATCH --output=results/logs/fac_backfill_%j.out
#SBATCH --error=results/logs/fac_backfill_%j.err
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh

# 1) Upload the cfgm_6/w5 results already generated
bash scripts/publish_to_fac.sh

# 2) Backfill figures for cfgm_9 across every cached weight
VARIANT_DIR=$(ls -d results/generated/cfg_matrix/cfgm_9_* 2>/dev/null | head -1)
VARIANT=$(basename "$VARIANT_DIR")
echo "cfgm_9 variant dir: $VARIANT_DIR"

for WDIR in "$VARIANT_DIR"/w*; do
  W=$(basename "$WDIR" | sed 's/^w//')
  echo "=== $VARIANT w$W ==="
  $PY scripts/evaluate_final.py --mode ptau217_mlp --dataset final --cond-mode ptau217 \
      --use-mentor-split --use-mask --use-cached --glass-brain \
      --unet-channels 128,256,512 --n-transformer 1 \
      --generated-dir "results/generated/cfg_matrix/${VARIANT}/w${W}" \
      --figures-dir "cfg_matrix/${VARIANT}/w${W}" \
      --records-dir "results/records/cfg_matrix/${VARIANT}/fold_0/w${W}"
done

# 3) Upload cfgm_9 results too
bash scripts/publish_to_fac.sh

# 4) Verify counts on FAC
FAC_DB=/mnt/fac/CX500024_DS2/ADNI_Data_INR/Diffusion_Baseline
for v in 6 9; do
  echo "cfgm_$v: $(find "$FAC_DB/results/figures/cfg_matrix" -path "*cfgm_${v}_*" -name '*.png' 2>/dev/null | wc -l) PNGs"
done
