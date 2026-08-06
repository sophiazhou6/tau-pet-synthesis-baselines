#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --mem=64G --time=04:00:00
#SBATCH --output=/home/remote/adebnath1/taugennet/results/logs/valcv_%x_%A_%a.out
#SBATCH --error=/home/remote/adebnath1/taugennet/results/logs/valcv_%x_%A_%a.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
F=$SLURM_ARRAY_TASK_ID
: ${ARM:?}; : ${CFG:?}; : ${LATENT:?}; : ${CONDEVAL:?}
CKPT=results/checkpoints/arch_sweep/$ARM/$CFG/fold_$F/diff_spatial_atrophy_best.pt
BASE=valcv/$ARM/$CFG/fold_$F/w1
G=results/generated/$BASE; REC=results/records/$BASE; mkdir -p $G $REC
FACO=$FACOUT/valcv/${ARM}_${CFG}/fold_$F
mkdir -p "$FACO" 2>/dev/null
LOG=/tmp/valcv_${ARM}_${CFG}_f$F.log
echo "== $ARM/$CFG fold $F ($(date)) ckpt=$CKPT ==" > $LOG
if [ ! -f "$CKPT" ]; then echo "MISSING CHECKPOINT" >> $LOG; timeout 60 cp $LOG "$FACO/fold_$F.log" 2>/dev/null; exit 1; fi
# --- generate on VAL split ---
$PY scripts/generate_spatial_sweep.py --checkpoint "$CKPT" --generated-dir $G \
    --latent-ch $LATENT --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 \
    --guidance-scale 1 --k 1 --fold $F --n-folds 5 --use-mask --use-mentor-split --eval-on val >> $LOG 2>&1
echo "-- generate done ($(date)) --" >> $LOG
timeout 60 cp $LOG "$FACO/fold_$F.log" 2>/dev/null   # EARLY publish so val-count is verifiable
# --- eval on VAL split ---
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode $CONDEVAL --fold $F --n-folds 5 \
    --use-cached --generated-dir $G --use-mask --use-mentor-split --skip-ablation --no-glass-brain \
    --eval-on val --records-dir $REC --metrics-out $REC/metrics.json >> $LOG 2>&1
echo "-- eval done ($(date)) --" >> $LOG
timeout 60 cp $LOG "$FACO/fold_$F.log" 2>/dev/null
timeout 90 cp $REC/pooled_wholebrain_*.csv $REC/region_pooled_*.csv $REC/region_crosssubject_metrics_*.csv $REC/metrics.json "$FACO/" 2>/dev/null
echo "fold $F DONE" >> $LOG; timeout 60 cp $LOG "$FACO/fold_$F.log" 2>/dev/null
