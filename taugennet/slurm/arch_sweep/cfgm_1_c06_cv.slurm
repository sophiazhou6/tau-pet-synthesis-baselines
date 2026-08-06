#!/bin/bash
#SBATCH --job-name=sw_cfgm_1_c06
#SBATCH --partition=gpu
#SBATCH --array=0-4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --mem=64G --time=08:00:00
#SBATCH --output=results/logs/sw_cfgm_1_c06_%A_%a.out
#SBATCH --error=results/logs/sw_cfgm_1_c06_%A_%a.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
F=$SLURM_ARRAY_TASK_ID
BASE=arch_sweep/cfgm_1/c06/fold_$F
OUT=results/checkpoints/$BASE; REC=results/records/$BASE; mkdir -p $OUT $REC results/logs; set -e
$PY scripts/train_spatial_sweep.py --cond-mode none --cfg-prob 0.15 --fold $F --n-folds 5 \
    --use-mentor-split --use-mask --latent-ch 4 --unet-channels 256,512,768 --n-transformer 3 \
    --noise-schedule linear --skip-ae --ae-checkpoint results/checkpoints/sweep/ae_lch4/fold_0/taugennet_checkpoint_best.pt \
    --diff-epochs 2000 --patience 250 --val-every 5 --checkpoint-dir $OUT
CKPT=$(ls -t $OUT/*best*.pt | head -1)
for W in 1 5 7 10 15 20; do
  G=results/generated/$BASE/w$W; mkdir -p $G
  $PY scripts/generate_spatial_sweep.py --checkpoint "$CKPT" --generated-dir $G \
      --latent-ch 4 --unet-channels 256,512,768 --n-transformer 3 --n-steps 500 \
      --guidance-scale $W --k 1 --fold $F --n-folds 5 --use-mask --use-mentor-split
  $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode atrophy --fold $F --n-folds 5 \
      --use-cached --generated-dir $G --use-mask --use-mentor-split --skip-ablation --no-glass-brain \
      --figures-dir $BASE/w$W --records-dir $REC/w$W --metrics-out $REC/w$W/metrics.json || echo "w$W failed"
done
bash scripts/publish_to_fac.sh
