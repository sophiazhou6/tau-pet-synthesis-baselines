#!/bin/bash
#SBATCH --job-name=ff_d_kl1e-6
#SBATCH --partition=gpu
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=64G --time=08:00:00
#SBATCH --output=results/logs/ff_d_kl1e-6_%j.out
#SBATCH --error=results/logs/ff_d_kl1e-6_%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet   # EDIT
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
AE=results/checkpoints/sweep/ae_lch3_kl1e-6/fold_0/taugennet_checkpoint_best.pt
OUT=results/checkpoints/sweep/diff_kl1e-6/fold_${SLURM_ARRAY_TASK_ID}; GEN=results/generated/sweep/kl1e-6/fold_${SLURM_ARRAY_TASK_ID}; REC=results/records/sweep/kl1e-6/fold_${SLURM_ARRAY_TASK_ID}
mkdir -p $OUT $GEN $REC; set -e
$PY scripts/train_cfg.py --mode combined_mlp --use-mentor-split --skip-ae --ae-checkpoint $AE \
    --unet-channels 128,256,512 --n-transformer 1 --cfg-prob 0.15 --diff-epochs 2000 --patience 250 --checkpoint-dir $OUT
$PY scripts/evaluate_cfg.py --mode combined_mlp --checkpoint-dir $OUT --use-best --use-mentor-split \
    --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 --use-mask --guidance-scale 10 --k 1 \
    --generated-dir $GEN --records-dir $REC --skip-ablation
$PY scripts/peak_from_gens.py --dataset combined --mode combined_mlp --use-mentor-split $GEN

bash scripts/publish_to_fac.sh
