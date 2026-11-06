#!/bin/bash
#SBATCH --job-name=cfg_ment
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G --time=10:00:00
#SBATCH --output=results/logs/cfg_ment_%j.out
#SBATCH --error=results/logs/cfg_ment_%j.err

# CFG-trained model on the MENTOR split, combined conditioning (atrophy MLP + CLIP p-tau).
# This is the controlled counterpart to b3 (combined + K16, NO CFG, mentor): same conditioning,
# same paper AE, same split -> isolates the effect of CFG. Evaluated with guidance w=10, K=16.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
AE=results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt
OUT=results/checkpoints/cfg_mentor/combined/fold_0
GEN=results/generated/cfg_mentor/combined_w10_k16/fold_0
REC=results/records/cfg_mentor/combined_w10_k16/fold_0
mkdir -p "$OUT" "$GEN" "$REC" results/logs
set -e

echo "=== train CFG (cfg_prob 0.15), combined, mentor split, paper AE, small UNet ==="
$PY scripts/train_cfg.py --mode combined --use-mentor-split \
    --skip-ae --ae-checkpoint "$AE" \
    --unet-channels 128,256,512 --n-transformer 1 \
    --cfg-prob 0.15 --diff-epochs 2000 --patience 250 \
    --checkpoint-dir "$OUT"

echo "=== evaluate: guidance w=10, K=16 posterior mean, MENTOR split ==="
$PY scripts/evaluate_cfg.py --mode combined \
    --checkpoint-dir "$OUT" --use-best --use-mentor-split \
    --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 --use-mask \
    --guidance-scale 10 --k 1 \
    --generated-dir "$GEN" \
    --figures-dir cfg_mentor/combined_w10_k16 \
    --records-dir "$REC"

echo "=== localization + all Pearsons ==="
$PY scripts/peak_from_gens.py --dataset combined --mode combined --use-mentor-split \
    "$GEN"
echo "Done: $(date)"

bash scripts/publish_to_fac.sh
