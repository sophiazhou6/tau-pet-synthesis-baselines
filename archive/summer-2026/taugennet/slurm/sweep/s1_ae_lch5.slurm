#!/bin/bash
#SBATCH --job-name=sw_ae5
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-18]
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/sw_ae5_%j.out
#SBATCH --error=results/logs/sw_ae5_%j.err

# STAGE 1a: AE for latent_ch=5. Paper spec: scale=8 (4 levels), 2 res-blocks/level, SiLU.
# batch-size 4 is REQUIRED: rb=2 doubles full-res activations (rb=2 @ bs=8 OOMs an 80GB A100).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet                   # EDIT for other server
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
OUT=results/checkpoints/sweep/ae_lch5/fold_0
mkdir -p $OUT results/logs
set -e
$PY scripts/train.py --mode ptau217 --fold 0 --n-folds 5 --use-mentor-split --use-mask \
    --ae-epochs 500 --diff-epochs 0 --batch-size 4 \
    --latent-ch 5 --latent-scale 8 --ae-res-blocks 2 --kl-weight 1e-5 \
    --patience 50 --ae-val-every 5 \
    --ae-checkpoint $OUT/taugennet_checkpoint.pt --checkpoint-dir $OUT
$PY scripts/eval_ae_recon.py --checkpoint $OUT/taugennet_checkpoint_best.pt \
    --mode ptau217 --fold 0 --n-folds 5 --split val --use-mask --no-figures \
    --metrics-out $OUT/recon_metrics.json
rm -f $OUT/taugennet_checkpoint.pt
echo "AE lch=5 done: $(date)"

bash scripts/publish_to_fac.sh
