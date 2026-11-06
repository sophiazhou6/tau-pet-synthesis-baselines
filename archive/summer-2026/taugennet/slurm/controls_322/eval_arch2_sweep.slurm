#!/bin/bash
#SBATCH --job-name=eval_arch2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/logs/eval_arch2_%j.out
#SBATCH --error=results/logs/eval_arch2_%j.err
set -uo pipefail
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
PY=/usr/bin/python3.9
# TAG:MAP:COND:LCH:CH:NT
COMBOS="\
arch_t1_atrenc_c02:atrophy:none:2:256,512,768:3 \
arch_t1_atrenc_c03:atrophy:none:3:128,256,512:1 \
arch_t1_atrenc_c04:atrophy:none:3:256,512,768:3 \
arch_t2b_both_c02:both:none:2:256,512,768:3 \
arch_t2b_both_c03:both:none:3:128,256,512:1 \
arch_t2b_both_c04:both:none:3:256,512,768:3 \
arch_t3_suvrencplasma_c02:suvr:ptau217:2:256,512,768:3 \
arch_t3_suvrencplasma_c03:suvr:ptau217:3:128,256,512:1 \
arch_t3_suvrencplasma_c04:suvr:ptau217:3:256,512,768:3"
for C in $COMBOS; do
  TAG=$(echo $C|cut -d: -f1); MAP=$(echo $C|cut -d: -f2); CM=$(echo $C|cut -d: -f3)
  LCH=$(echo $C|cut -d: -f4); CH=$(echo $C|cut -d: -f5); NT=$(echo $C|cut -d: -f6)
  export TAUGENNET_SPATIAL_MAP=$MAP
  for F in 0 1 2 3 4; do
    CK=$(ls -t results/checkpoints/controls_322/$TAG/fold_$F/*best*.pt 2>/dev/null | head -1)
    if [ -z "$CK" ]; then echo "SKIP $TAG f$F"; continue; fi
    GD=results/generated/controls_322_arch2/$TAG/fold_$F
    RD=results/records/controls_322_arch2/$TAG/fold_$F
    mkdir -p "$GD" "$RD"
    echo "=== $TAG fold=$F map=$MAP cond=$CM lch=$LCH $(date) ==="
    $PY scripts/generate_spatial.py --checkpoint "$CK" --generated-dir "$GD" \
        --use-controls-322 --eval-on val --fold $F --n-folds 5 \
        --latent-ch $LCH --unet-channels $CH --n-transformer $NT --n-steps 500 --use-mask
    if [ $? -ne 0 ]; then echo "GEN FAILED $TAG f$F"; continue; fi
    $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode $CM \
        --use-controls-322 --eval-on val --fold $F --n-folds 5 --use-cached \
        --generated-dir "$GD" --records-dir "$RD" \
        --figures-dir silu/controls_322_arch2/$TAG/fold_$F \
        --use-mask --no-glass-brain --skip-ablation
  done
done
echo "=== ARCH2 EVAL DONE $(date) ==="
