#!/bin/bash
# Usage: ./gen_stage23.sh <winning_latent_ch> <small|large>
# Stage 2 = kl sweep (AE hyperparameter -> needs a fresh AE per value; 1e-5 already done in stage 1).
# Stage 3 = noise schedule (diffusion-only -> reuses the winning AE; linear already done).
set -e
LCH=$1; ARM=$2
[ -z "$ARM" ] && { echo "usage: $0 <latent_ch> <small|large>"; exit 1; }
if [ "$ARM" = "small" ]; then CH="128,256,512"; NT=1; else CH="256,512,768"; NT=3; fi
cd "$(dirname "$0")/../.."

for KL in 1e-3 1e-4 1e-6; do          # 1e-5 is the stage-1 winner cell, already trained
  cat > slurm/sweep/s2_ae_kl${KL}.slurm << INNER
#!/bin/bash
#SBATCH --job-name=sw_ae_kl${KL}
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G --time=06:00:00
#SBATCH --output=results/logs/sw_ae_kl${KL}_%j.out
#SBATCH --error=results/logs/sw_ae_kl${KL}_%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/sz3962/.conda/envs/taugennet/bin/python3
cd \$SLURM_SUBMIT_DIR
OUT=results/checkpoints/sweep/ae_lch${LCH}_kl${KL}/fold_0
mkdir -p \$OUT results/logs
set -e
\$PY scripts/train_cfg.py --mode $COND_MODE --use-mentor-split --use-mask \\
    --ae-epochs 500 --diff-epochs 0 --batch-size 4 \\
    --latent-ch ${LCH} --latent-scale 8 --ae-res-blocks 2 --kl-weight ${KL} \\
    --patience 50 --ae-val-every 5 \\
    --ae-checkpoint \$OUT/taugennet_checkpoint.pt --checkpoint-dir \$OUT
rm -f \$OUT/taugennet_checkpoint.pt
INNER

  cat > slurm/sweep/s2_diff_kl${KL}.slurm << INNER
#!/bin/bash
#SBATCH --job-name=sw_d_kl${KL}
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G --time=08:00:00
#SBATCH --output=results/logs/sw_d_kl${KL}_%j.out
#SBATCH --error=results/logs/sw_d_kl${KL}_%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/sz3962/.conda/envs/taugennet/bin/python3
cd \$SLURM_SUBMIT_DIR
AE=results/checkpoints/sweep/ae_lch${LCH}_kl${KL}/fold_0/taugennet_checkpoint_best.pt
OUT=results/checkpoints/sweep/diff_kl${KL}/fold_0
GEN=results/generated/sweep/kl${KL}/fold_0
REC=results/records/sweep/kl${KL}/fold_0
mkdir -p \$OUT \$GEN \$REC
set -e
\$PY scripts/train_cfg.py --mode $COND_MODE --use-mentor-split --use-mask \\
    --latent-ch ${LCH} --unet-channels ${CH} --n-transformer ${NT} \\
    --skip-ae --ae-checkpoint \$AE \\
    --cfg-prob 0.15 --diff-epochs 2000 --patience 250 --checkpoint-dir \$OUT
\$PY scripts/generate_posterior_mean.py --mode ptau217 --checkpoint-dir \$OUT --use-best \\
    --k 1 --n-steps 500 --unet-channels ${CH} --n-transformer ${NT} --arch silu \\
    --fold 0 --n-folds 5 --use-mentor-split --use-mask --out-dir \$GEN
\$PY scripts/evaluate_final.py --mode ptau217 --dataset final --fold 0 --n-folds 5 \\
    --use-cached --generated-dir \$GEN --use-mask --use-mentor-split --skip-ablation \\
    --figures-dir sweep/kl${KL} --records-dir \$REC --metrics-out \$REC/metrics.json
\$PY scripts/localization_metrics.py \$REC --sort-by peak
INNER
done

# Stage 3: cosine only (linear already trained). Diffusion-only -> reuse the winning stage-1 AE.
cat > slurm/sweep/s3_diff_cosine.slurm << INNER
#!/bin/bash
#SBATCH --job-name=sw_cosine
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --mem=64G --time=08:00:00
#SBATCH --output=results/logs/sw_cosine_%j.out
#SBATCH --error=results/logs/sw_cosine_%j.err
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/sz3962/.conda/envs/taugennet/bin/python3
cd \$SLURM_SUBMIT_DIR
AE=results/checkpoints/sweep/ae_lch${LCH}/fold_0/taugennet_checkpoint_best.pt
OUT=results/checkpoints/sweep/diff_cosine/fold_0
GEN=results/generated/sweep/cosine/fold_0
REC=results/records/sweep/cosine/fold_0
mkdir -p \$OUT \$GEN \$REC
set -e
\$PY scripts/train_cfg.py --mode $COND_MODE --use-mentor-split --use-mask \\
    --latent-ch ${LCH} --unet-channels ${CH} --n-transformer ${NT} --noise-schedule cosine \\
    --skip-ae --ae-checkpoint \$AE \\
    --cfg-prob 0.15 --diff-epochs 2000 --patience 250 --checkpoint-dir \$OUT
\$PY scripts/generate_posterior_mean.py --mode ptau217 --checkpoint-dir \$OUT --use-best \\
    --k 1 --n-steps 500 --unet-channels ${CH} --n-transformer ${NT} --arch silu \\
    --fold 0 --n-folds 5 --use-mentor-split --use-mask --out-dir \$GEN
\$PY scripts/evaluate_final.py --mode ptau217 --dataset final --fold 0 --n-folds 5 \\
    --use-cached --generated-dir \$GEN --use-mask --use-mentor-split --skip-ablation \\
    --figures-dir sweep/cosine --records-dir \$REC --metrics-out \$REC/metrics.json
\$PY scripts/localization_metrics.py \$REC --sort-by peak
INNER
echo "generated stage2 (3 AE + 3 diff) and stage3 (1 diff) for latent_ch=${LCH} arm=${ARM}"
