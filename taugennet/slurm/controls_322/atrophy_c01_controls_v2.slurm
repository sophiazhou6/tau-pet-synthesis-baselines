#!/bin/bash
#SBATCH --job-name=ctrl322_atrophy_c01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --output=/home/remote/adebnath1/taugennet/results/logs/ctrl322_atrophy_c01_%j.out
#SBATCH --error=/home/remote/adebnath1/taugennet/results/logs/ctrl322_atrophy_c01_%j.err
cd ~/taugennet
export TAUGENNET_ROOT=$HOME/taugennet
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
LOGFILE="/home/remote/adebnath1/taugennet/results/logs/ctrl322_atrophy_c01_${SLURM_JOB_ID}.out"
FAC_HEARTBEAT="/mnt/fac/CX500024_DS2/ADNI_Data_INR/Diffusion_Baseline/taugennet_deliverables/_cluster_out/ctrl322_atrophy_c01_progress.txt"
( while true; do
    sleep 900
    { echo "=== heartbeat $(date), job $SLURM_JOB_ID ==="; tail -30 "$LOGFILE" 2>/dev/null; } > /tmp/ctrl322_heartbeat.txt
    timeout 15 cp /tmp/ctrl322_heartbeat.txt "$FAC_HEARTBEAT" 2>/dev/null
  done ) &
HEARTBEAT_PID=$!

/usr/bin/python3.9 scripts/train_cfg.py --mode atrophy --use-controls-322 \
    --skip-ae --ae-checkpoint results/checkpoints/sweep/ae_lch2/fold_0/taugennet_checkpoint_best.pt \
    --unet-channels 128,256,512 --n-transformer 1 \
    --diff-epochs 2000 --patience 250 --val-every 5 \
    --checkpoint-dir results/checkpoints/controls_322/atrophy_c01
TRAIN_RC=$?

kill "$HEARTBEAT_PID" 2>/dev/null
{ echo "=== FINAL $(date), job $SLURM_JOB_ID, exit code $TRAIN_RC ==="; tail -60 "$LOGFILE" 2>/dev/null; } > /tmp/ctrl322_heartbeat.txt
timeout 15 cp /tmp/ctrl322_heartbeat.txt "$FAC_HEARTBEAT" 2>/dev/null
