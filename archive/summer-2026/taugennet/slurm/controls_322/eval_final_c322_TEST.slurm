#!/bin/bash
#SBATCH --job-name=eval_final_c322_TEST
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=results/logs/eval_final_c322_TEST_%j.out
#SBATCH --error=results/logs/eval_final_c322_TEST_%j.err

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"

echo "=== atrophy_c01 (small arch) — $(date) ==="
$PYTHON scripts/evaluate_final.py --mode atrophy \
    --checkpoint-dir results/checkpoints/controls_322/atrophy_c01 --use-best \
    --use-controls-322 --eval-on test \
    --unet-channels 128,256,512 --n-transformer 1 \
    --figures-dir silu/controls_322_TEST/atrophy_c01 \
    --records-dir results/records/controls_322_TEST/atrophy_c01 \
    --generated-dir results/generated/controls_322_TEST/atrophy_c01 \
    --arch silu --use-mask --n-steps 500 --glass-brain --save-generated
echo "atrophy_c01 done: $(date)"

echo "=== combinedmlp_c01 (small arch) — $(date) ==="
$PYTHON scripts/evaluate_final.py --mode combined_mlp \
    --checkpoint-dir results/checkpoints/controls_322/combinedmlp_c01 --use-best \
    --use-controls-322 --eval-on test \
    --unet-channels 128,256,512 --n-transformer 1 \
    --figures-dir silu/controls_322_TEST/combinedmlp_c01 \
    --records-dir results/records/controls_322_TEST/combinedmlp_c01 \
    --generated-dir results/generated/controls_322_TEST/combinedmlp_c01 \
    --arch silu --use-mask --n-steps 500 --glass-brain --save-generated
echo "combinedmlp_c01 done: $(date)"

echo "=== combinedmlp_c02 (large arch) — $(date) ==="
$PYTHON scripts/evaluate_final.py --mode combined_mlp \
    --checkpoint-dir results/checkpoints/controls_322/combinedmlp_c02 --use-best \
    --use-controls-322 --eval-on test \
    --unet-channels 256,512,768 --n-transformer 3 \
    --figures-dir silu/controls_322_TEST/combinedmlp_c02 \
    --records-dir results/records/controls_322_TEST/combinedmlp_c02 \
    --generated-dir results/generated/controls_322_TEST/combinedmlp_c02 \
    --arch silu --use-mask --n-steps 500 --glass-brain --save-generated
echo "combinedmlp_c02 done: $(date)"

echo "=== suvrcond_c01 (small arch) — $(date) ==="
$PYTHON scripts/evaluate_final.py --mode atrophy --suvr-as-cond \
    --checkpoint-dir results/checkpoints/controls_322/suvrcond_c01 --use-best \
    --use-controls-322 --eval-on test \
    --unet-channels 128,256,512 --n-transformer 1 \
    --figures-dir silu/controls_322_TEST/suvrcond_c01 \
    --records-dir results/records/controls_322_TEST/suvrcond_c01 \
    --generated-dir results/generated/controls_322_TEST/suvrcond_c01 \
    --arch silu --use-mask --n-steps 500 --glass-brain --save-generated
echo "suvrcond_c01 done: $(date)"

echo "All controls_322 canonical evals complete: $(date)"
