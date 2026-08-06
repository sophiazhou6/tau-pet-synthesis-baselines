#!/bin/bash
#SBATCH --job-name=sw_dSP
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --mem=64G --time=08:00:00
#SBATCH --output=results/logs/sw_dSP_%j.out
#SBATCH --error=results/logs/sw_dSP_%j.err
# SPATIAL-family sweep cell (use if the CFG matrix winner is a spatial-MAP conditioning).
# Copy per latent_ch/arm; set LCH, CH/NT, and COND_MODE (none|ptau217|ptau217_mlp).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet                   # EDIT
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
LCH=3; CH=128,256,512; NT=1; COND_MODE=ptau217         # <-- SET
AE=results/checkpoints/sweep/ae_lch${LCH}/fold_0/taugennet_checkpoint_best.pt
OUT=results/checkpoints/sweep/diffSP_lch${LCH}/fold_0
GEN=results/generated/sweep/SP_lch${LCH}/fold_0
REC=results/records/sweep/SP_lch${LCH}/fold_0
mkdir -p $OUT $GEN $REC results/logs
set -e
$PY scripts/train_spatial_atrophy.py --cond-mode $COND_MODE --cfg-prob 0.15 \
    --fold 0 --n-folds 5 --use-mentor-split --use-mask \
    --latent-ch $LCH --unet-channels $CH --n-transformer $NT \
    --skip-ae --ae-checkpoint $AE --diff-epochs 2000 --patience 250 --checkpoint-dir $OUT
CKPT=$(ls -t $OUT/*best*.pt | head -1)
$PY scripts/generate_spatial.py --checkpoint "$CKPT" --generated-dir $GEN \
    --latent-ch $LCH --unet-channels $CH --n-transformer $NT \
    --n-steps 500 --guidance-scale 10 --k 1 --fold 0 --n-folds 5 --use-mask --use-mentor-split
E=$([ "$COND_MODE" = "none" ] && echo atrophy || echo ptau217)
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode $E \
    --fold 0 --n-folds 5 --use-cached --generated-dir $GEN --use-mask --use-mentor-split --skip-ablation \
    --records-dir $REC --metrics-out $REC/metrics.json
$PY scripts/peak_from_gens.py --dataset spatial --mode atrophy --cond-mode $E --use-mentor-split $GEN

bash scripts/publish_to_fac.sh
