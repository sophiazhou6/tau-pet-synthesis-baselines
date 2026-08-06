#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/final_%x_%j.out
#SBATCH --error=results/logs/final_%x_%j.err
MAP=$1; CM=$2; TAG=$3; EPOCHS=$4; EXTRA=${5:-}
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
export TAUGENNET_SPATIAL_MAP=$MAP
PY=/usr/bin/python3.9
CKDIR=results/checkpoints/controls_322/$TAG

echo "=== FINAL TRAIN $TAG  map=$MAP cond=$CM epochs=$EPOCHS (all dev, no val) $(date) ==="
$PY scripts/train_spatial_atrophy.py --use-controls-322 --cond-mode $CM \
    --no-val-split --diff-epochs $EPOCHS --patience 999999 --val-every 999999 \
    --skip-ae --ae-checkpoint results/checkpoints/sweep/ae_lch2/fold_0/taugennet_checkpoint_best.pt \
    --latent-ch 2 --ae-res-blocks 2 --unet-channels 128,256,512 --n-transformer 1 \
    --checkpoint-dir $CKDIR --use-mask $EXTRA
if [ $? -ne 0 ]; then echo "TRAIN FAILED"; exit 1; fi

CK=$(ls -t $CKDIR/*.pt 2>/dev/null | head -1)
echo "=== HELD-OUT TEST EVAL  ckpt=$CK  $(date) ==="
GD=results/generated/controls_322_TEST/$TAG
RD=results/records/controls_322_TEST/$TAG
mkdir -p "$GD" "$RD"
$PY scripts/generate_spatial.py --checkpoint "$CK" --generated-dir "$GD" \
    --use-controls-322 --eval-on test --latent-ch 2 \
    --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 --use-mask
if [ $? -ne 0 ]; then echo "GEN FAILED"; exit 1; fi
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode $CM \
    --use-controls-322 --eval-on test --use-cached \
    --generated-dir "$GD" --records-dir "$RD" \
    --figures-dir silu/controls_322_TEST/$TAG \
    --use-mask --glass-brain --skip-ablation
echo "=== DONE $TAG $(date) ==="
