#!/bin/bash
#SBATCH --job-name=phase6_canary
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=results/logs/phase6_canary_%j.out
set -x
set -eo pipefail
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
export TAUGENNET_ROOT=$HOME/taugennet
cd $HOME/taugennet
PY=/usr/bin/python3.9
F=0
OUT=results/checkpoints/phase6_canary/fold_$F
mkdir -p $OUT
echo "=== PHASE6 CANARY: combined_mlp (atrophy86 + ptau1) + ATROPHY spatial map, pooled AD+MCI+CN, 2 epochs ==="
echo "=== watch for: '[phase6] ... matched=235' , 'cond_vec dim = 87' , train/val/test counts ==="
$PY scripts/train_spatial_sweep.py --cond-mode ptau217_mlp --phase6-manifest results/records/phase6_manifest_v4.csv --cfg-prob 0.15 --fold $F --n-folds 5 \
    --use-mask --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 \
    --noise-schedule cosine --skip-ae \
    --ae-checkpoint results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt \
    --diff-epochs 2 --patience 1 --val-every 1 --checkpoint-dir $OUT
CKPT=$(ls -t $OUT/*.pt 2>/dev/null | head -1)
[ -n "$CKPT" ] && echo "[phase6 canary] TRAINING PASS — checkpoint: $CKPT" || { echo "[phase6 canary] FAIL — no checkpoint"; exit 1; }
