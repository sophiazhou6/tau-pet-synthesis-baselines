#!/bin/bash
#SBATCH --job-name=sw_d6s
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --mem=64G --time=08:00:00
#SBATCH --output=results/logs/sw_d6_small_%j.out
#SBATCH --error=results/logs/sw_d6_small_%j.err
# STAGE 1b: diffusion, latent_ch=6, UNet=small (128,256,512, nt=1), WITH CFG.
# COND_MODE = the winning conditioning from the CFG matrix. Default 'combined' (atrophy MLP+CLIP);
# set to combined_mlp / atrophy / ptau217_mlp as appropriate. (Final-family modes only — for a
# spatial-MAP winner use the s1_diff_SPATIAL_template.slurm instead.)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet                   # EDIT
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
COND_MODE=combined_mlp                                     # <-- SET to the CFG-matrix winner
AE=results/checkpoints/sweep/ae_lch6/fold_0/taugennet_checkpoint_best.pt
OUT=results/checkpoints/sweep/diff_lch6_small/fold_0
GEN=results/generated/sweep/lch6_small/fold_0
REC=results/records/sweep/lch6_small/fold_0
mkdir -p $OUT $GEN $REC results/logs
set -e
# CFG training (conditioning-dropout) + CFG+posterior-mean eval (w=10, K=16 = best recipe)
$PY scripts/train_cfg.py --mode $COND_MODE --use-mentor-split --skip-ae --ae-checkpoint $AE \
    --unet-channels 128,256,512 --n-transformer 1 --cfg-prob 0.15 \
    --diff-epochs 2000 --patience 250 --checkpoint-dir $OUT
$PY scripts/evaluate_cfg.py --mode $COND_MODE --checkpoint-dir $OUT --use-best --use-mentor-split \
    --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 --use-mask \
    --guidance-scale 10 --k 1 --generated-dir $GEN --records-dir $REC
# rank on ALL FIVE: SET1 / SET2 / SET3 / WITHIN / PEAK
DS=$([ "$COND_MODE" = "atrophy" ] && echo final || echo combined)
$PY scripts/peak_from_gens.py --dataset $DS --mode $COND_MODE --use-mentor-split $GEN

bash scripts/publish_to_fac.sh
