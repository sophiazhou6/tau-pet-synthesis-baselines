#!/bin/bash
#SBATCH --job-name=ctrl322_combmlp_c02
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=ggpu1-[06-12]
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/remote/adebnath1/taugennet/results/logs/ctrl322_combmlp_c02_%j.out
#SBATCH --error=/home/remote/adebnath1/taugennet/results/logs/ctrl322_combmlp_c02_%j.err
cd ~/taugennet
export TAUGENNET_ROOT=$HOME/taugennet
export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
/usr/bin/python3.9 scripts/train_cfg.py --mode combined_mlp --use-controls-322 \
    --skip-ae --ae-checkpoint results/checkpoints/sweep/ae_lch2/fold_0/taugennet_checkpoint_best.pt \
    --unet-channels 256,512,768 --n-transformer 3 \
    --diff-epochs 2000 --patience 250 --val-every 5 \
    --checkpoint-dir results/checkpoints/controls_322/combinedmlp_c02
