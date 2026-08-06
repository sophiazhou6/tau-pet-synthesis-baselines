#!/bin/bash
#SBATCH --job-name=vconf
#SBATCH --partition=gpu
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 --mem=64G --time=01:30:00
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --output=/home/remote/adebnath1/taugennet/results/logs/vconf_%A_%a.out
#SBATCH --error=/home/remote/adebnath1/taugennet/results/logs/vconf_%A_%a.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
CONFIGS=("cfgm_1 c03 3 atrophy" "cfgm_5 c03 3 ptau217")
CI=$((SLURM_ARRAY_TASK_ID / 5)); F=$((SLURM_ARRAY_TASK_ID % 5))
read ARM CFG L COND <<< "${CONFIGS[$CI]}"
CKPT=results/checkpoints/arch_sweep/$ARM/$CFG/fold_$F/diff_spatial_atrophy_best.pt
G=results/generated/valconfirm/$ARM/$CFG/fold_$F/w1; REC=results/records/valconfirm/$ARM/$CFG/fold_$F/w1
mkdir -p $G $REC
FACO=/mnt/fac/CX500024_DS2/ADNI_Data_INR/Diffusion_Baseline/taugennet_deliverables/_cluster_out/valconfirm/${ARM}_${CFG}/fold_$F
mkdir -p "$FACO"
LOG=/tmp/vconf_${ARM}_${CFG}_f$F.log
echo "== $ARM/$CFG fold $F ($(date)) L=$L cond=$COND ==" > $LOG
[ -f "$CKPT" ] || { echo "MISSING CKPT" >> $LOG; timeout 60 cp $LOG "$FACO/f$F.log"; exit 1; }
$PY scripts/generate_spatial_sweep.py --checkpoint "$CKPT" --generated-dir $G \
   --latent-ch $L --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 \
   --guidance-scale 1 --k 1 --fold $F --n-folds 5 --use-mask --use-mentor-split --eval-on val >> $LOG 2>&1
echo "-- gen done --" >> $LOG; timeout 60 cp $LOG "$FACO/f$F.log" 2>/dev/null
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode $COND --fold $F --n-folds 5 \
   --use-cached --generated-dir $G --use-mask --use-mentor-split --skip-ablation --no-glass-brain \
   --eval-on val --records-dir $REC --metrics-out $REC/metrics.json >> $LOG 2>&1
echo "-- eval done --" >> $LOG
timeout 60 cp $LOG "$FACO/f$F.log" 2>/dev/null
timeout 90 cp $REC/pooled_wholebrain_*.csv $REC/region_pooled_*.csv $REC/region_crosssubject_metrics_*.csv $REC/metrics.json "$FACO/" 2>/dev/null
