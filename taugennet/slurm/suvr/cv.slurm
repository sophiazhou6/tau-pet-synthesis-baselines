#!/bin/bash
#SBATCH --job-name=suvr_cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --array=0-4
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --output=results/logs/suvr_cv_%A_%a.out
set -x
set -eo pipefail
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
export TAUGENNET_ROOT=$HOME/taugennet
export TAUGENNET_SPATIAL_MAP=suvr          # paint regional SUVR into the spatial channel
cd $HOME/taugennet
PY=/usr/bin/python3.9
F=$SLURM_ARRAY_TASK_ID
BASE=suvr_spatial/fold_$F
OUT=results/checkpoints/$BASE
REC=results/records/$BASE
mkdir -p $OUT $REC
echo "=== SUVR 5-fold CV: fold $F | c13 config (lch3 cosine small) + SUVR spatial map ==="
$PY scripts/train_spatial_sweep.py --cond-mode ptau217_mlp --cfg-prob 0.15 --fold $F --n-folds 5 \
    --use-mentor-split --use-mask --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 \
    --noise-schedule cosine --skip-ae \
    --ae-checkpoint results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt \
    --diff-epochs 2000 --patience 250 --val-every 5 --checkpoint-dir $OUT
CKPT=$(ls -t $OUT/*best*.pt | head -1)
for W in 1 5; do
  G=results/generated/$BASE/w$W; mkdir -p $G
  $PY scripts/generate_spatial_sweep.py --checkpoint "$CKPT" --generated-dir $G \
      --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 \
      --guidance-scale $W --k 1 --fold $F --n-folds 5 --use-mask --use-mentor-split
  $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --fold $F --n-folds 5 \
      --use-cached --generated-dir $G --use-mask --use-mentor-split --skip-ablation --no-glass-brain \
      --figures-dir $BASE/w$W --records-dir $REC/w$W --metrics-out $REC/w$W/metrics.json
done
bash scripts/publish_to_fac.sh
echo "[suvr cv] fold $F done"
