#!/bin/bash
#SBATCH --job-name=biobert_ptau217
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/biobert_ptau217_%j.out
#SBATCH --error=results/logs/biobert_ptau217_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

# BiomedBERT (PubMedBERT) p-tau217 conditioning — A/B vs the CLIP ptau217 run.
# Same AE, same hyperparams as slurm/train/train_silu_500ep.slurm; only the text
# encoder differs (mode = ptau217_biobert). Isolated checkpoint/figure dirs.
#
# PREREQ: run scripts/download_biomedbert.py on the LOGIN NODE first so
# biomedbert_model/ exists (GPU nodes are offline).

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
AE_CKPT=results/checkpoints/taugennet_checkpoint_silu_masked.pt

cd /scratch/network/sz3962/taugennet

mkdir -p results/logs \
         results/checkpoints/biobert_ptau217 \
         results/figures/monitor_biobert_ptau217

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "=== ptau217_biobert (500 ep) ==="
$PYTHON scripts/train.py \
    --mode ptau217_biobert \
    --skip-ae \
    --ae-checkpoint $AE_CKPT \
    --diff-epochs 500 \
    --patience 600 \
    --val-every 1 \
    --monitor-every 25 \
    --monitor-dir results/figures/monitor_biobert_ptau217 \
    --checkpoint-dir results/checkpoints/biobert_ptau217 \
    --use-mask
echo "ptau217_biobert done at: $(date)"

echo "Run complete. End time: $(date)"
