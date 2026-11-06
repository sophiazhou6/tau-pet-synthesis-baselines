#!/bin/bash
#SBATCH --job-name=final_c04
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --output=results/logs/final_c04_%j.out
set -x
set -eo pipefail
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
export TAUGENNET_ROOT=$HOME/taugennet
cd $HOME/taugennet
PY=/usr/bin/python3.9
: "${E:?must pass --export=ALL,E=<epochs>}"
TAG=e${E}
OUT=results/checkpoints/final/cfgm_5_c04_$TAG
G=results/generated/final/cfgm_5_c04_$TAG/w1
REC=results/records/final/cfgm_5_c04_$TAG/w1
mkdir -p $OUT $G $REC
echo "=== FINAL: c04 (lch3 LARGE), all-dev no-val, ${E} epochs ==="

$PY scripts/train_spatial_sweep.py --cond-mode ptau217_mlp --cfg-prob 0.15 --fold 0 --n-folds 5 \
    --use-mentor-split --use-mask --no-val-split \
    --latent-ch 3 --unet-channels 256,512,768 --n-transformer 3 --noise-schedule linear \
    --skip-ae --ae-checkpoint results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt \
    --diff-epochs $E --patience 999999 --val-every 999999 --monitor-every 999999 --checkpoint-dir $OUT

ls -la $OUT
CKPT=$OUT/diff_spatial_atrophy.pt
[ -f "$CKPT" ] || { echo "!! no checkpoint at $CKPT"; ls -la $OUT; exit 1; }

$PY scripts/generate_spatial_sweep.py --checkpoint "$CKPT" --generated-dir $G \
    --latent-ch 3 --unet-channels 256,512,768 --n-transformer 3 --n-steps 500 \
    --guidance-scale 1 --k 1 --fold 0 --n-folds 5 --use-mask --use-mentor-split

$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 \
    --fold 0 --n-folds 5 --use-cached --generated-dir $G --use-mask --use-mentor-split \
    --skip-ablation --glass-brain \
    --figures-dir final/cfgm_5_c04_$TAG/w1 --records-dir $REC --metrics-out $REC/metrics.json

bash scripts/publish_to_fac.sh
