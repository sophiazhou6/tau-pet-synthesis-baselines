#!/bin/bash
#SBATCH --job-name=cfgm_2_atrophyMLP
#SBATCH --partition=gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --mem=64G --time=08:00:00
#SBATCH --output=results/logs/cfgm_2_atrophyMLP_%j.out
#SBATCH --error=results/logs/cfgm_2_atrophyMLP_%j.err
#SBATCH --mail-type=END,FAIL --mail-user=sz3962@princeton.edu
# CFG mode=atrophy, small UNet, mentor, paper AE. Trains once; evaluates a W-SWEEP at K=1.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/usr/bin/python3.9
cd /home/remote/adebnath1/taugennet
export PYTHONPATH="/home/remote/adebnath1/.local/lib/python3.9/site-packages:/home/remote/adebnath1/taugennet:$PYTHONPATH"
export TAUGENNET_ROOT="/home/remote/adebnath1/taugennet"
source scripts/wait_for_fac.sh
OUT=results/checkpoints/cfg_matrix/cfgm_2_atrophyMLP/fold_${SLURM_ARRAY_TASK_ID}; REC=results/records/cfg_matrix/cfgm_2_atrophyMLP/fold_${SLURM_ARRAY_TASK_ID}
mkdir -p $OUT $REC results/logs; set -e
$PY scripts/train_cfg.py --mode atrophy --use-mentor-split --skip-ae --ae-checkpoint results/checkpoints/paper_aligned/ae/fold_0/taugennet_checkpoint_best.pt \
    --unet-channels 128,256,512 --n-transformer 1 --cfg-prob 0.15 --diff-epochs 2000 --patience 250 --checkpoint-dir $OUT
DS=$([ "atrophy" = "atrophy" ] && echo final || echo combined)
for W in 1 5 7 10 15 20; do
  $PY scripts/evaluate_cfg.py --mode atrophy --checkpoint-dir $OUT --use-best --use-mentor-split \
      --unet-channels 128,256,512 --n-transformer 1 --n-steps 500 --use-mask \
      --guidance-scale $W --k 1 --generated-dir results/generated/cfg_matrix/cfgm_2_atrophyMLP/fold_${SLURM_ARRAY_TASK_ID}/w$W --skip-ablation \
      --figures-dir cfg_matrix/cfgm_2_atrophyMLP/w$W --records-dir $REC/w$W || echo "w$W failed"
done
echo "=== cfgm_2_atrophyMLP : SET1/2/3 + WITHIN + PEAK across w ==="
$PY scripts/peak_from_gens.py --dataset $DS --mode atrophy --use-mentor-split results/generated/cfg_matrix/cfgm_2_atrophyMLP/fold_${SLURM_ARRAY_TASK_ID}/w*

bash scripts/publish_to_fac.sh
