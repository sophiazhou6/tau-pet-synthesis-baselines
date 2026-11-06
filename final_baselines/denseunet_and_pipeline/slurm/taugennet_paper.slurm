#!/bin/bash
#SBATCH --job-name=tg_paper
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=adroit-h11g2
#SBATCH --mem=96G
#SBATCH --output=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/tg_paper_%j.out
#SBATCH --error=/scratch/network/sz3962/tau-denseunet-baseline/results/logs/tg_paper_%j.err
#SBATCH --mail-type=FAIL
#
# TauGenNet as published (Gong et al., IEEE TRPMS; arXiv 2509.04269) on cohort_v2, plasma p-tau217
# through frozen CLIP ("Plasma is X.XXX."), missing plasma MICE-imputed per split
# (scripts/impute_plasma.py). Architecture = the summer paper-aligned spec (latent 3, AE 2 res-blocks
# per level, U-Net 256/512/768 with 3 transformer layers); diffusion Adam lr 1e-4, batch 8, a fixed
# 600 epochs, final checkpoint; DDPM 500 steps (respaced). Scored with the same GitHub pipeline as the
# Dense-U-Net (scripts/score_baseline.py).
#
#   STAGE=ae   SPLIT=<k>  sbatch --time=... slurm/taugennet_paper.slurm   AE, early stopping on fold k's val
#   STAGE=diff SPLIT=<k>  sbatch ...   diffusion (600 ep) on fold k's best AE → generate + score val and test
#   STAGE=ae   SPLIT=refit sbatch ...   AE on all 477, no val, length = median fold best epoch (or AE_EPOCHS)
#   STAGE=diff SPLIT=refit sbatch ...   diffusion on all 477 → generate + score the held-out test
#   STAGE=smoke sbatch --time=03:00:00 ...   tiny end-to-end run of both paths (tag taugennet_smoke)
# Fold diffusion checkpoints (2.6 GB) are deleted once their predictions are scored; the refit's is kept.
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/sz3962/.conda/envs/taugennet/bin/python3
GH=/scratch/network/sz3962/tau-pet-synthesis-baselines/taugennet
REPO=/scratch/network/sz3962/tau-denseunet-baseline
# the conda env has an editable install mapping `src` to the OLD local copy (/scratch/network/sz3962/
# taugennet/src); `python scripts/x.py` does not put the cwd on sys.path, so without this the GitHub
# scripts would import that copy (no imputed plasma, no respaced sampler). PYTHONPATH wins over it.
export PYTHONPATH=$GH${PYTHONPATH:+:$PYTHONPATH}
STAGE="${STAGE:?set STAGE=ae|diff|smoke}"
TAG="${TAG:-taugennet_paper}"
UNET_ARGS="--unet-channels 256,512,768 --n-transformer 3"     # latent_ch / AE arch come from the checkpoint
AE_ARGS="--batch-size 4 --latent-ch 3 --latent-scale 8 --ae-res-blocks 2 --kl-weight 1e-5"
SCRATCH_TMP="${TMPDIR:-/tmp}/tg_$SLURM_JOB_ID"; mkdir -p "$SCRATCH_TMP"
trap 'rm -rf "$SCRATCH_TMP"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_split() {   # $1 = fold index or "refit"; $2 = ae epochs; $3 = diffusion epochs; $4 = sampler steps; $5 = stages
  local split=$1 ae_ep=$2 diff_ep=$3 nsteps=$4 stages=" $5 "
  local sub split_args eval_args ae_file roles
  if [[ "$split" == refit ]]; then
    sub=single; split_args="--split 80/0/20"; eval_args=""; roles="test"
  else
    sub=fold_$split; split_args="--fold $split --n-folds 5"; eval_args="$split_args"; roles="val test"
  fi
  export TAUGENNET_ROOT=$GH TAUGENNET_PLASMA_CSV=$GH/data/raw/plasma_imputed/$([[ $split == refit ]] && echo refit || echo fold_$split).csv
  local ck=$GH/results/checkpoints/cohort_v2/$TAG/$sub
  mkdir -p "$ck/ae" "$ck/diff"
  cd "$GH"
  echo "── $TAG $sub: stages [$5]  plasma $TAUGENNET_PLASMA_CSV  ($(date)) ──"

  if [[ "$stages" == *" ae "* ]]; then
    if [[ "$split" == refit ]]; then        # fixed length from the CV folds, no validation set
      $PY scripts/train.py --mode ptau217 --use-controls-322 $split_args --use-mask $AE_ARGS \
          --ae-epochs "$ae_ep" --diff-epochs 0 \
          --ae-checkpoint "$ck/ae/taugennet_checkpoint.pt" --checkpoint-dir "$ck/ae" --figures-dir "$ck/ae/figs"
    else                                    # early stopping on the fold's validation subjects
      $PY scripts/train.py --mode ptau217 --use-controls-322 $split_args --use-mask $AE_ARGS \
          --ae-epochs "$ae_ep" --diff-epochs 0 --patience 50 --ae-val-every "${AE_VAL_EVERY:-5}" \
          --ae-checkpoint "$ck/ae/taugennet_checkpoint.pt" --checkpoint-dir "$ck/ae" --figures-dir "$ck/ae/figs"
      rm -f "$ck/ae/taugennet_checkpoint.pt"            # keep the best-on-val AE only
    fi
  fi

  if [[ "$stages" == *" diff "* ]]; then
    if [[ "$split" == refit ]]; then ae_file=$ck/ae/taugennet_checkpoint.pt
    else ae_file=$ck/ae/taugennet_checkpoint_best.pt; fi
    if [[ ! -f "$ae_file" ]]; then echo "no AE checkpoint $ae_file"; exit 1; fi
    # fixed budget, final checkpoint: validation and the SSIM monitor off (no 2.6 GB best-on-val copies)
    $PY scripts/train.py --mode ptau217 --use-controls-322 $split_args --use-mask --latent-ch 3 $UNET_ARGS \
        --skip-ae --ae-checkpoint "$ae_file" --diff-epochs "$diff_ep" --batch-size 8 --diff-weight-decay 0 \
        --patience 1000000 --val-every 1000000 --monitor-every 1000000 \
        --checkpoint-dir "$ck/diff" --figures-dir "$ck/diff/figs"
  fi

  if [[ "$stages" == *" score "* ]]; then
    for role in $roles; do
      local sfx="" fold_args=""
      if [[ "$role" == val ]]; then sfx=_val; fi
      if [[ "$split" != refit ]]; then fold_args="--fold $split"; fi
      local gen=$REPO/results/generated/baselines/cohort_v2/$TAG/$sub$sfx
      rm -rf "$gen"; mkdir -p "$gen"
      $PY scripts/evaluate_final.py --mode ptau217 --dataset final --checkpoint-dir "$ck/diff" $UNET_ARGS \
          --arch silu $eval_args --use-controls-322 --use-mask --save-generated --generated-dir "$gen" \
          --n-steps "$nsteps" --eval-on "$role" --skip-ablation --no-glass-brain \
          --figures-dir "$SCRATCH_TMP/fig_$role" --records-dir "$SCRATCH_TMP/rec_$role" \
          --metrics-out "$SCRATCH_TMP/rec_$role/metrics.json"
      # generation order = the GitHub loader's order for this split (what gh_align checks against)
      $PY - "$gen" "$role" $split <<'IDX'
import re, sys
import pandas as pd
sys.path.insert(0, ".")
from src.dataset_final import build_dataloaders
gen, role, split = sys.argv[1], sys.argv[2], sys.argv[3]
kw = dict(use_dk_mask=True, use_controls_322=True)
kw.update(dict(train_frac=0.8, val_frac=0.0) if split == "refit" else dict(fold_idx=int(split), n_folds=5))
_, va, te, *_ = build_dataloaders("ptau217", **kw)
ds = va if role == "val" else te
rids = [re.search(r"RID[_-](\d+)", p).group(1) for p in ds.pet_paths]
pd.DataFrame({"index": range(len(rids)), "RID": rids}).to_csv(f"{gen}/index.csv", index=False)
print(f"[index] {role}: {len(rids)} subjects → {gen}/index.csv")
IDX
      (cd "$REPO" && TAUGENNET_ROOT=/scratch/network/sz3962/taugennet $PY scripts/score_baseline.py \
          --manifest cohort_v2 --tag "$TAG" $fold_args --split "$role" --use-controls-322 --generated-dir "$gen")
    done
    # fold model scored → free 2.6 GB (KEEP_FOLD_DIFF=1 keeps it, for a re-run still to be checked)
    if [[ "$split" != refit && -z "${KEEP_FOLD_DIFF:-}" ]]; then rm -f "$ck/diff/diff_ptau217.pt"; fi
  fi
  echo "── $TAG $sub done ($(date)) ──"
}

fold_ae_median() {   # refit AE length = median best-on-validation epoch of the 5 fold AEs
  $PY - "$GH/results/checkpoints/cohort_v2/$TAG" <<'MED'
import sys, numpy as np, torch
base = sys.argv[1]
ep = [int(torch.load(f"{base}/fold_{k}/ae/taugennet_checkpoint_best.pt", map_location="cpu", weights_only=False)["epoch"])
      for k in range(5)]
print(f"fold AE best epochs {ep}", file=sys.stderr)
print(int(np.median(ep)))
MED
}

case "$STAGE" in
  ae)    if [[ "${SPLIT:?}" == refit && -z "${AE_EPOCHS:-}" ]]; then AE_EPOCHS=$(fold_ae_median); fi
         run_split "$SPLIT" "${AE_EPOCHS:-500}" 0 500 "ae" ;;
  diff)  run_split "${SPLIT:?}" 0 "${DIFF_EPOCHS:-600}" 500 "diff score" ;;
  smoke) TAG=taugennet_smoke; AE_VAL_EVERY=1
         # code-path test on any GPU: the AE at batch 4 needs an 80 GB A100 (OOM on a 46 GB A40);
         # diffusion keeps batch 8, so the smoke also shows whether the diff jobs fit an A40
         AE_ARGS="${AE_ARGS/--batch-size 4/--batch-size 1}"
         if [[ " ${SMOKE_SPLITS:-0 refit} " == *" 0 "* ]]; then run_split 0 2 2 10 "ae diff score"; fi
         if [[ " ${SMOKE_SPLITS:-0 refit} " == *" refit "* ]]; then run_split refit 1 1 10 "ae diff score"; fi
         rm -rf "$GH/results/checkpoints/cohort_v2/taugennet_smoke"
         echo "SMOKE OK" ;;
  *)     echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
