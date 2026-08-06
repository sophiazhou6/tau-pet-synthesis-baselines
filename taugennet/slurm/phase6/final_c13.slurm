#!/bin/bash
#SBATCH --job-name=phase6_final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=results/logs/phase6_final_%j.out
set -x
set -eo pipefail
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
export TAUGENNET_ROOT=$HOME/taugennet
cd $HOME/taugennet
PY=/usr/bin/python3.9
MANIFEST=results/records/phase6_manifest_v4.csv
EPOCHS=${EPOCHS:-520}
NSTEPS=${NSTEPS:-500}
TAG=${TAG:-c13_ptau_e${EPOCHS}}
OUT=results/checkpoints/phase6_final/$TAG
REC=results/records/phase6_final/$TAG
mkdir -p $OUT $REC
echo "=== PHASE6 1.2: c13 (lch3 cosine) + ptau217_mlp + ATROPHY map, pooled 235, no-val-split, ${EPOCHS} epochs ==="

# --- train on all dev (val folded in), no early stopping ---
$PY scripts/train_spatial_sweep.py --cond-mode ptau217_mlp --phase6-manifest $MANIFEST --cfg-prob 0.15 \
    --fold 0 --n-folds 5 --use-mask --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 \
    --noise-schedule cosine --skip-ae --ae-checkpoint results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt \
    --no-val-split --diff-epochs $EPOCHS --patience 999999 --val-every 999999 --monitor-every 999999 \
    --checkpoint-dir $OUT

CKPT=$(ls -t $OUT/*.pt 2>/dev/null | head -1)
[ -n "$CKPT" ] || { echo "FAIL: no checkpoint saved (no-val-split save issue)"; exit 1; }
echo "checkpoint: $CKPT"

# --- generate + evaluate at guidance w1 and w5 (SET1/SET3 + glass brains) ---
for W in 1 5; do
  G=results/generated/phase6_final/$TAG/w$W; mkdir -p $G
  $PY scripts/generate_spatial_sweep.py --checkpoint "$CKPT" --generated-dir $G \
      --phase6-manifest $MANIFEST --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 \
      --noise-schedule cosine --n-steps $NSTEPS --guidance-scale $W --k 1 --fold 0 --n-folds 5 --use-mask
  $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 \
      --phase6-manifest $MANIFEST --fold 0 --n-folds 5 --use-cached --generated-dir $G \
      --use-mask --skip-ablation --glass-brain \
      --figures-dir phase6_final/$TAG/w$W --records-dir $REC/w$W --metrics-out $REC/w$W/metrics.json
done
FAC_DB=/mnt/fac/CX500024_DS2/ADNI_Data_INR/Diffusion_Baseline
if [ -d "$FAC_DB" ]; then
  mkdir -p "$FAC_DB/results/records" "$FAC_DB/results/figures" "$FAC_DB/results/checkpoints"
  rsync -a results/records/phase6_final "$FAC_DB/results/records/"
  rsync -a results/figures/phase6_final "$FAC_DB/results/figures/"
  rsync -a results/checkpoints/phase6_final "$FAC_DB/results/checkpoints/"
  echo "[phase6 1.2] synced to FAC"
fi
echo "[phase6 1.2] DONE"
