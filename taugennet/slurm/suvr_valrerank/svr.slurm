#!/bin/bash
#SBATCH --job-name=svr
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 --mem=64G --time=01:30:00
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --output=/home/remote/adebnath1/taugennet/results/logs/svr_%A_%a.out
#SBATCH --error=/home/remote/adebnath1/taugennet/results/logs/svr_%A_%a.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TAUGENNET_SPATIAL_MAP=suvr
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
F=$SLURM_ARRAY_TASK_ID
CKPT=$(ls -t results/checkpoints/suvr_spatial/fold_$F/*best*.pt | head -1)
G=results/generated/suvr_valrerank/fold_$F/w1; REC=results/records/suvr_valrerank/fold_$F/w1
mkdir -p $G $REC
FACO=/mnt/fac/CX500024_DS2/ADNI_Data_INR/Diffusion_Baseline/taugennet_deliverables/_cluster_out/suvr_valrerank/fold_$F
mkdir -p "$FACO"
LOG=/tmp/svr_f$F.log
echo "== suvr val fold $F ($(date)) ckpt=$CKPT ==" > $LOG
[ -f "$CKPT" ] || { echo "MISSING CKPT" >> $LOG; timeout 60 cp $LOG "$FACO/log.txt"; exit 1; }
$PY scripts/generate_spatial_sweep.py --checkpoint "$CKPT" --generated-dir $G \
   --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 \
   --guidance-scale 1 --k 1 --fold $F --n-folds 5 --use-mask --use-mentor-split --eval-on val >> $LOG 2>&1
timeout 60 cp $LOG "$FACO/log.txt" 2>/dev/null
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --fold $F --n-folds 5 \
   --use-cached --generated-dir $G --use-mask --use-mentor-split --skip-ablation --no-glass-brain \
   --eval-on val --records-dir $REC --metrics-out $REC/metrics.json >> $LOG 2>&1
echo "-- done $(date) --" >> $LOG
timeout 60 cp $LOG "$FACO/log.txt" 2>/dev/null
timeout 90 cp $REC/pooled_wholebrain_*.csv $REC/region_pooled_*.csv $REC/region_crosssubject_metrics_*.csv $REC/metrics.json "$FACO/" 2>/dev/null
