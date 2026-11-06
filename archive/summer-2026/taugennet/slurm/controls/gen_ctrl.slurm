#!/bin/bash
#SBATCH --job-name=ctrl_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4 --mem=64G --time=01:30:00
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --output=/home/remote/adebnath1/taugennet/results/logs/ctrl_full_%j.out
#SBATCH --error=/home/remote/adebnath1/taugennet/results/logs/ctrl_full_%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MENTOR_TEST_CSV=heldout_test_split_control.csv
export MENTOR_DEV_CSV=train_val_split_control.csv
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
CKPT=results/checkpoints/arch_sweep/cfgm_1/c01/fold_0/diff_spatial_atrophy_best.pt
G=results/generated/controls_full/cfgm_1_c01; REC=results/records/controls_full/cfgm_1_c01
mkdir -p $G $REC
FACO=/mnt/fac/CX500024_DS2/ADNI_Data_INR/Diffusion_Baseline/taugennet_deliverables/_cluster_out/controls_full
mkdir -p "$FACO"
LOG=/tmp/ctrl_full.log
echo "== controls FULL $(date) test=$MENTOR_TEST_CSV ==" > $LOG
$PY scripts/generate_spatial_sweep.py --checkpoint "$CKPT" --generated-dir $G \
   --latent-ch 2 --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 \
   --guidance-scale 1 --k 1 --use-mask --use-mentor-split --eval-on test >> $LOG 2>&1
echo "-- gen done $(date) --" >> $LOG; timeout 60 cp $LOG "$FACO/ctrl_full.log" 2>/dev/null
$PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode atrophy \
   --use-cached --generated-dir $G --use-mask --use-mentor-split --skip-ablation --no-glass-brain \
   --eval-on test --records-dir $REC --metrics-out $REC/metrics.json >> $LOG 2>&1
echo "-- eval done $(date) --" >> $LOG
timeout 60 cp $LOG "$FACO/ctrl_full.log" 2>/dev/null
timeout 90 cp $REC/pooled_wholebrain_*.csv $REC/region_pooled_*.csv $REC/region_crosssubject_metrics_*.csv $REC/metrics.json "$FACO/" 2>/dev/null
