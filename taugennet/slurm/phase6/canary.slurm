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
G=results/generated/phase6_canary/fold_$F/w1
REC=results/records/phase6_canary/fold_$F/w1
mkdir -p $OUT $G $REC
echo "=== SUVR CANARY: c13 config (lch3 cosine small) + SUVR spatial map, 2 epochs ==="
echo "=== watch for: 'SUVR subjects: N' and 'Matched subjects (...): M' (M should be > 0) ==="
$PY scripts/train_spatial_sweep.py --cond-mode combined_mlp --phase6-manifest results/records/phase6_manifest_v4.csv --cfg-prob 0.15 --fold $F --n-folds 5 \
    --use-mask --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 \
    --noise-schedule cosine --skip-ae \
    --ae-checkpoint results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt \
    --diff-epochs 2 --patience 1 --val-every 1 --checkpoint-dir $OUT
CKPT=$(ls -t $OUT/*best*.pt | head -1)
$PY scripts/generate_spatial_sweep.py --checkpoint "$CKPT" --generated-dir $G \
    --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 --n-steps 10 \
    --guidance-scale 1 --k 1 --fold $F --n-folds 5 --use-mask --use-mentor-split
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --fold $F --n-folds 5 \
    --use-cached --generated-dir $G --use-mask --skip-ablation --no-glass-brain \
    --figures-dir phase6_canary/fold_$F/w1 --records-dir $REC --metrics-out $REC/metrics.json
echo "[suvr canary] PASS"
