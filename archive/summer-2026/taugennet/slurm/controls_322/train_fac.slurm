#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=results/logs/fac_%x_%j.out
#SBATCH --error=results/logs/fac_%x_%j.err
MAP=$1; CM=$2; LCH=$3; AE=$4; CH=$5; NT=$6; TAG=$7; FOLD=$8; EXTRA=${9:-}
[ "$EXTRA" = "-" ] && EXTRA=""
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
export TAUGENNET_SPATIAL_MAP=$MAP
echo "=== $TAG fold=$FOLD map=$MAP cond=$CM lch=$LCH ch=$CH nt=$NT extra='$EXTRA' $(date) ==="
/usr/bin/python3.9 scripts/train_spatial_atrophy.py \
    --use-controls-322 --cond-mode $CM --fold $FOLD --n-folds 5 \
    --skip-ae --ae-checkpoint $AE --latent-ch $LCH --ae-res-blocks 2 \
    --unet-channels $CH --n-transformer $NT \
    --checkpoint-dir results/checkpoints/controls_322/$TAG/fold_$FOLD \
    --diff-epochs 2000 --patience 500 --use-mask $EXTRA
echo "exit=$? done: $(date)"
