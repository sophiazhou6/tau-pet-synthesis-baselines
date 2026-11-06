#!/bin/bash
#SBATCH --job-name=bf_cfgm8_figs
#SBATCH --partition=cpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=03:00:00
#SBATCH --output=results/logs/bf_cfgm8_%j.out
#SBATCH --error=results/logs/bf_cfgm8_%j.err
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
REC=results/records/cfg_matrix/cfgm_8_ptauCLIP/fold_0
for W in 1 5 7 10 15 20; do
  echo "=== w$W ==="
  $PY scripts/evaluate_final.py --mode ptau217 --dataset final --cond-mode ptau217 \
      --use-mentor-split --use-mask --use-cached --glass-brain \
      --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 \
      --generated-dir results/generated/cfg_matrix/cfgm_8_ptauCLIP/w$W \
      --figures-dir cfg_matrix/cfgm_8_ptauCLIP/w$W --records-dir $REC/w$W || echo "w$W failed"
done
bash scripts/publish_to_fac.sh
