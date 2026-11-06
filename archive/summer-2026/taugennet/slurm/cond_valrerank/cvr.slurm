#!/bin/bash
#SBATCH --job-name=cvr
#SBATCH --partition=gpu
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 --mem=64G --time=01:00:00
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --output=/home/remote/adebnath1/taugennet/results/logs/cvr_%A_%a.out
#SBATCH --error=/home/remote/adebnath1/taugennet/results/logs/cvr_%A_%a.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
ARMS=(
"cfgm_1_atrophySPATIAL spatial atrophy"
"cfgm_4_atrSPAT_ptauCLIP spatial ptau217"
"cfgm_5_atrSPAT_ptauMLP spatial ptau217"
"cfgm_7_tissue spatial ptau217"
"cfgm_2_atrophyMLP cfg atrophy"
"cfgm_6_atrMLP_ptauMLP cfg combined_mlp"
"cfgm_8_ptauCLIP cfg ptau217"
"cfgm_9_ptauMLP cfg ptau217_mlp"
)
read ARM TYPE M <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"
OUT=results/checkpoints/cfg_matrix/$ARM/fold_0
G=results/generated/cond_valrerank/$ARM/w1; REC=results/records/cond_valrerank/$ARM/w1
mkdir -p $G $REC
FACO=/mnt/fac/CX500024_DS2/ADNI_Data_INR/Diffusion_Baseline/taugennet_deliverables/_cluster_out/cond_valrerank/$ARM
mkdir -p "$FACO"
LOG=/tmp/cvr_$ARM.log
echo "== $ARM type=$TYPE mode=$M ($(date)) ==" > $LOG
if [ "$TYPE" = "spatial" ]; then
  CKPT=$(ls -t $OUT/*best*.pt | head -1); echo "ckpt=$CKPT" >> $LOG
  $PY scripts/generate_spatial.py --checkpoint "$CKPT" --generated-dir $G \
     --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 \
     --k 1 --fold 0 --n-folds 5 --use-mask --use-mentor-split --eval-on val >> $LOG 2>&1
  timeout 60 cp $LOG "$FACO/log.txt" 2>/dev/null
  $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode $M --fold 0 --n-folds 5 \
     --use-cached --generated-dir $G --use-mask --use-mentor-split --skip-ablation --no-glass-brain \
     --eval-on val --records-dir $REC --metrics-out $REC/metrics.json >> $LOG 2>&1
else
  $PY scripts/evaluate_cfg.py --mode $M --checkpoint-dir $OUT --use-best --use-mentor-split \
     --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 --use-mask \
     --guidance-scale 1 --k 1 --generated-dir $G --skip-ablation --records-dir $REC --eval-on val >> $LOG 2>&1
fi
echo "-- done $(date) --" >> $LOG
timeout 60 cp $LOG "$FACO/log.txt" 2>/dev/null
timeout 90 cp $REC/pooled_wholebrain_*.csv $REC/region_pooled_*.csv $REC/region_crosssubject_metrics_*.csv $REC/metrics.json "$FACO/" 2>/dev/null
