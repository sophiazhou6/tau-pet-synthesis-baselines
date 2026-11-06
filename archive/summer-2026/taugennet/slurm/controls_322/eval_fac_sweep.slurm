#!/bin/bash
#SBATCH --job-name=eval_fac_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=results/logs/eval_fac_sweep_%j.out
#SBATCH --error=results/logs/eval_fac_sweep_%j.err
set -uo pipefail
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
PY=/usr/bin/python3.9
COMBOS="fac_suvrmap_c02:suvr:none:2:256,512,768:3 fac_suvrmap_c03:suvr:none:3:128,256,512:1 fac_suvrmap_c04:suvr:none:3:256,512,768:3 fac_suvratr_c02:suvr:atrophy:2:256,512,768:3 fac_suvratr_c03:suvr:atrophy:3:128,256,512:1 fac_suvratr_c04:suvr:atrophy:3:256,512,768:3 fac_atrmap_c02:atrophy:none:2:256,512,768:3 fac_atrmap_c03:atrophy:none:3:128,256,512:1 fac_atrmap_c04:atrophy:none:3:256,512,768:3 fac_suvrenc_c02:suvr:none:2:256,512,768:3 fac_suvrenc_c03:suvr:none:3:128,256,512:1 fac_suvrenc_c04:suvr:none:3:256,512,768:3 fac_atratr_c02:atrophy:atrophy:2:256,512,768:3 fac_atratr_c03:atrophy:atrophy:3:128,256,512:1 fac_atratr_c04:atrophy:atrophy:3:256,512,768:3"
for C in $COMBOS; do
  TAG=$(echo $C|cut -d: -f1); MAP=$(echo $C|cut -d: -f2); CM=$(echo $C|cut -d: -f3)
  LCH=$(echo $C|cut -d: -f4); CH=$(echo $C|cut -d: -f5); NT=$(echo $C|cut -d: -f6)
  export TAUGENNET_SPATIAL_MAP=$MAP
  for F in 0 1 2 3 4; do
    CK=$(ls -t results/checkpoints/controls_322/$TAG/fold_$F/*best*.pt 2>/dev/null | head -1)
    if [ -z "$CK" ]; then echo "SKIP $TAG f$F"; continue; fi
    GD=results/generated/controls_322_fac/$TAG/fold_$F
    RD=results/records/controls_322_fac/$TAG/fold_$F
    echo "=== $TAG fold=$F $(date) ==="
    $PY scripts/generate_spatial.py --checkpoint "$CK" --generated-dir "$GD" \
        --use-controls-322 --eval-on val --fold $F --n-folds 5 \
        --latent-ch $LCH --unet-channels $CH --n-transformer $NT --n-steps 500 --use-mask
    if [ $? -ne 0 ]; then echo "GEN FAILED $TAG f$F"; continue; fi
    $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode $CM \
        --use-controls-322 --eval-on val --fold $F --n-folds 5 --use-cached \
        --generated-dir "$GD" --records-dir "$RD" \
        --figures-dir silu/controls_322_fac/$TAG/fold_$F \
        --use-mask --no-glass-brain --skip-ablation
    echo "$TAG fold $F done $(date)"
  done
done
echo "=== FACTORIAL SWEEP DONE $(date) ==="
