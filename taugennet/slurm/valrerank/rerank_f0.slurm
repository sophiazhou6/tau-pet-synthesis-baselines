#!/bin/bash
#SBATCH --job-name=rerank
#SBATCH --partition=gpu
#SBATCH --array=0-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 --mem=64G --time=01:00:00
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --output=/home/remote/adebnath1/taugennet/results/logs/rerank_%A_%a.out
#SBATCH --error=/home/remote/adebnath1/taugennet/results/logs/rerank_%A_%a.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
CONFIGS=(
"cfgm_5 c01 2 128,256,512 1 ptau217"
"cfgm_5 c02 2 256,512,768 3 ptau217"
"cfgm_5 c03 3 128,256,512 1 ptau217"
"cfgm_5 c04 3 256,512,768 3 ptau217"
"cfgm_5 c05 4 128,256,512 1 ptau217"
"cfgm_5 c06 4 256,512,768 3 ptau217"
"cfgm_5 c07 5 128,256,512 1 ptau217"
"cfgm_5 c08 5 256,512,768 3 ptau217"
"cfgm_5 c09 6 128,256,512 1 ptau217"
"cfgm_5 c10 6 256,512,768 3 ptau217"
"cfgm_1 c02 2 256,512,768 3 atrophy"
"cfgm_1 c03 3 128,256,512 1 atrophy"
"cfgm_1 c04 3 256,512,768 3 atrophy"
)
read ARM CFG L CH NT COND <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
F=0
CKPT=results/checkpoints/arch_sweep/$ARM/$CFG/fold_$F/diff_spatial_atrophy_best.pt
G=results/generated/valrerank/$ARM/$CFG/fold_$F/w1; REC=results/records/valrerank/$ARM/$CFG/fold_$F/w1
mkdir -p $G $REC
FACO=/mnt/fac/CX500024_DS2/ADNI_Data_INR/Diffusion_Baseline/taugennet_deliverables/_cluster_out/valrerank/${ARM}_${CFG}
mkdir -p "$FACO"
LOG=/tmp/rerank_${ARM}_${CFG}.log
echo "== $ARM/$CFG fold $F ($(date)) L=$L CH=$CH NT=$NT cond=$COND ==" > $LOG
[ -f "$CKPT" ] || { echo "MISSING CKPT $CKPT" >> $LOG; timeout 60 cp $LOG "$FACO/f0.log"; exit 1; }
$PY scripts/generate_spatial_sweep.py --checkpoint "$CKPT" --generated-dir $G \
   --latent-ch $L --unet-channels $CH --n-transformer $NT --n-steps 500 \
   --guidance-scale 1 --k 1 --fold $F --n-folds 5 --use-mask --use-mentor-split --eval-on val >> $LOG 2>&1
echo "-- gen done $(date) --" >> $LOG; timeout 60 cp $LOG "$FACO/f0.log" 2>/dev/null
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode $COND --fold $F --n-folds 5 \
   --use-cached --generated-dir $G --use-mask --use-mentor-split --skip-ablation --no-glass-brain \
   --eval-on val --records-dir $REC --metrics-out $REC/metrics.json >> $LOG 2>&1
echo "-- eval done $(date) --" >> $LOG
timeout 60 cp $LOG "$FACO/f0.log" 2>/dev/null
timeout 90 cp $REC/pooled_wholebrain_*.csv $REC/region_pooled_*.csv $REC/region_crosssubject_metrics_*.csv $REC/metrics.json "$FACO/" 2>/dev/null
