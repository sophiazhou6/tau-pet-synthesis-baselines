#!/bin/bash
#SBATCH --job-name=rescore_ae_dk86
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --constraint=gpu80|v100
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=results/logs/rescore_ae_dk86_%j.out
#SBATCH --error=results/logs/rescore_ae_dk86_%j.err

# Re-score the existing latent_ch{4,8} × lp_weight ablation AEs (+ the lch8 baseline)
# with the CORRECTED DK86-masked recon metric, on the held-out test split.
# Each AE's latent_ch is read from its own checkpoint, so no --latent-ch needed.

PYTHON=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
OUT=results/records/ae_rescore_dk86
mkdir -p "$OUT" results/logs

echo "Job ID: $SLURM_JOB_ID  Node: $SLURM_NODELIST  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TAGS=(lch4_lp0.0 lch4_lp0.1 lch4_lp0.5 lch4_lp1.0 lch8_lp0.0 lch8_lp0.1 lch8_lp0.5 lch8_lp1.0 lch8_baseline_dk86)

for t in "${TAGS[@]}"; do
    if [ "$t" = "lch8_baseline_dk86" ]; then
        CK=results/checkpoints/ae_lch8_dk86/ae_checkpoint_best.pt
    else
        CK=results/checkpoints/ae_exp_masked_v2/$t/ae_checkpoint_best.pt
    fi
    echo "==== $t  ($CK) ===="
    $PYTHON scripts/eval_ae_recon.py \
        --checkpoint "$CK" \
        --mode atrophy --split test \
        --use-mask --no-figures \
        --metrics-out "$OUT/$t.json"
done

echo "Done: $(date)"
