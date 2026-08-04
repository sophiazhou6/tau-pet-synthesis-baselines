#!/bin/bash
#SBATCH --job-name=ae_lch_lp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-7
#SBATCH --output=results/logs/ae_lch_lp_%A_%a.out
#SBATCH --error=results/logs/ae_lch_lp_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sz3962@princeton.edu

PYTHON=/home/sz3962/.conda/envs/taugennet/bin/python3
cd /scratch/network/sz3962/taugennet

# Condition index → (latent_ch, lp_weight)
# 0-3: latent_ch=4 × lp ∈ {0.0, 0.1, 0.5, 1.0}
# 4-7: latent_ch=8 × lp ∈ {0.0, 0.1, 0.5, 1.0}
LATENT_CHS=(4 4 4 4 8 8 8 8)
LP_WEIGHTS=(0.0 0.1 0.5 1.0 0.0 0.1 0.5 1.0)

LATENT_CH=${LATENT_CHS[$SLURM_ARRAY_TASK_ID]}
LP_WEIGHT=${LP_WEIGHTS[$SLURM_ARRAY_TASK_ID]}
TAG="lch${LATENT_CH}_lp${LP_WEIGHT}"
OUTDIR="results/checkpoints/ae_exp_masked_v2/${TAG}"
FIGDIR="results/figures/ae_exp_masked_v2/${TAG}"

mkdir -p "$OUTDIR" "$FIGDIR" results/logs

echo "Array task:  $SLURM_ARRAY_TASK_ID  →  latent_ch=${LATENT_CH}  lp_weight=${LP_WEIGHT}"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Start time:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

$PYTHON scripts/train_lp_modality.py \
    --mode atrophy \
    --diff-epochs 0 \
    --ae-epochs 400 \
    --patience 50 \
    --latent-ch "${LATENT_CH}" \
    --lp-weight "${LP_WEIGHT}" \
    --modality-weight 0.0 \
    --ae-checkpoint "${OUTDIR}/ae_checkpoint.pt" \
    --checkpoint-dir "${OUTDIR}" \
    --figures-dir "${FIGDIR}"

echo "Done: $(date)"
