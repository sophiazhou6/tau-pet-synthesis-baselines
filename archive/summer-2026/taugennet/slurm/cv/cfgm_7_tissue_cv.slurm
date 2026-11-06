#!/bin/bash
#SBATCH --job-name=cfgm_7_tissue
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G --time=08:00:00
#SBATCH --output=results/logs/cfgm_7_tissue_%j.out
#SBATCH --error=results/logs/cfgm_7_tissue_%j.err
#SBATCH --mail-type=END,FAIL --mail-user=sz3962@princeton.edu
# CFG SPATIAL family: atrophy spatial map + TISSUE one-hot (GM/WM/CSF) + CLIP ptau, small UNet,
# mentor, paper AE. Small-arch + CFG = the FAIR tissue test (vs the old large-arch/no-CFG run).
# Train once; W-SWEEP {1,5,7,10,15,20} at K=1.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
OUT=results/checkpoints/cfg_matrix/cfgm_7_tissue/fold_${SLURM_ARRAY_TASK_ID}; REC=results/records/cfg_matrix/cfgm_7_tissue/fold_${SLURM_ARRAY_TASK_ID}
mkdir -p $OUT $REC results/logs; set -e
$PY scripts/train_spatial_atrophy.py --cond-mode ptau217 --use-tissue --cfg-prob 0.15 \
    --fold ${SLURM_ARRAY_TASK_ID} --n-folds 5 --use-mentor-split --use-mask \
    --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 \
    --skip-ae --ae-checkpoint results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt --diff-epochs 2000 --patience 250 --val-every 5 --checkpoint-dir $OUT
CKPT=$(ls -t $OUT/*best*.pt | head -1)
for W in 1 5 7 10 15 20; do
  G=results/generated/cfg_matrix/cfgm_7_tissue/fold_${SLURM_ARRAY_TASK_ID}/w$W; mkdir -p $G
  $PY scripts/generate_spatial.py --checkpoint "$CKPT" --generated-dir $G \
      --latent-ch 3 --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 \
      --guidance-scale $W --k 1 --fold ${SLURM_ARRAY_TASK_ID} --n-folds 5 --use-mask --use-mentor-split
  $PY scripts/evaluate_final.py --mode atrophy --dataset spatial --cond-mode ptau217 --fold ${SLURM_ARRAY_TASK_ID} --n-folds 5 \
      --use-cached --generated-dir $G --use-mask --use-mentor-split --skip-ablation \
      --figures-dir cfg_matrix/cfgm_7_tissue/w$W --records-dir $REC/w$W --metrics-out $REC/w$W/metrics.json || echo "w$W failed"
done
echo "=== cfgm_7_tissue : SET1/2/3 + WITHIN + PEAK across w ==="
$PY scripts/peak_from_gens.py --dataset spatial --mode atrophy --cond-mode ptau217 --use-mentor-split results/generated/cfg_matrix/cfgm_7_tissue/fold_${SLURM_ARRAY_TASK_ID}/w*

bash scripts/publish_to_fac.sh
