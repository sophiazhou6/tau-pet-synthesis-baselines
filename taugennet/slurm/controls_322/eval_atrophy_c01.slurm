#!/bin/bash
#SBATCH --job-name=eval_ctrl322_atrophy_c01
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/remote/adebnath1/taugennet/results/logs/eval_ctrl322_atrophy_c01_%j.out
#SBATCH --error=/home/remote/adebnath1/taugennet/results/logs/eval_ctrl322_atrophy_c01_%j.err
cd ~/taugennet
export TAUGENNET_ROOT=$HOME/taugennet
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
/usr/bin/python3.9 scripts/evaluate_cfg.py --mode atrophy --eval-on val --use-controls-322  \
    --checkpoint-dir results/checkpoints/controls_322/atrophy_c01 \
    --unet-channels 128,256,512 --n-transformer 1 \
    --records-dir results/records/controls_322_eval/atrophy_c01 \
    --figures-dir controls_322_eval/atrophy_c01
