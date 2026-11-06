#!/bin/bash
#SBATCH --job-name=eval_newcond
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=results/logs/eval_newcond_%j.out
#SBATCH --error=results/logs/eval_newcond_%j.err
set -uo pipefail
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
PY=/usr/bin/python3.9
ARMS="cv_atrophymap_learnedenc:atrophy:none cv_suvrenc_plasma:suvr:ptau217 cv_atrophyenc_plasma:atrophy:ptau217 cv_bothmaps_enc:both:none cv_bothmaps_enc_plasma:both:ptau217"
for A in $ARMS; do
  TAG=$(echo $A|cut -d: -f1); MAP=$(echo $A|cut -d: -f2); CM=$(echo $A|cut -d: -f3)
  export TAUGENNET_SPATIAL_MAP=$MAP
  for F in 0 1 2 3 4; do
    CK=$(ls -t results/checkpoints/controls_322/$TAG/fold_$F/*best*.pt 2>/dev/null | head -1)
    if [ -z "$CK" ]; then echo "SKIP $TAG f$F -- no checkpoint"; continue; fi
    GD=results/generated/controls_322_cv/$TAG/fold_$F
    RD=results/records/controls_322_cv/$TAG/fold_$F
    echo "=== $TAG fold=$F map=$MAP cond=$CM $(date) ==="
    $PY scripts/generate_spatial.py --checkpoint "$CK" --generated-dir "$GD" \
        --use-controls-322 --eval-on val --fold $F --n-folds 5 \
        --latent-ch 2 --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 --use-mask
    if [ $? -ne 0 ]; then echo "GEN FAILED $TAG f$F"; continue; fi
    $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode $CM \
        --use-controls-322 --eval-on val --fold $F --n-folds 5 --use-cached \
        --generated-dir "$GD" --records-dir "$RD" \
        --figures-dir silu/controls_322_cv/$TAG/fold_$F \
        --use-mask --glass-brain --skip-ablation
    echo "$TAG fold $F done $(date)"
  done
done
echo "=== NEW-COND SWEEP DONE $(date) ==="
