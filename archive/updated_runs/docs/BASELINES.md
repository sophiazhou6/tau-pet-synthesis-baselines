# Baseline model cards

Every model below is trained, generated and scored by the same three scripts on the same
manifest cohort, folds, inputs and metric code (see *Common protocol*). Per-model defaults are
in `baselines/registry.py`; each can be overridden on the `train_baseline.py` command line.

| name | family | source | generator | params |
|---|---|---|---|---|
| `denseunet` | deterministic regression | Lee et al., *Brain* 2024 (Neurology-AI-Program/AI_imputed_tau_PET) | 3D Dense-U-Net | 44.8 M |
| `swinunetr` | deterministic regression | Hatamizadeh et al., arXiv 2201.01266 (MONAI `SwinUNETR`) | Swin-Transformer encoder + CNN decoder | 62.2 M |
| `eagan` / `deagan` | 3D conditional GAN | Yu et al., *IEEE TMI* 2019 (by-lab/Ea-GANs, MIT) | 3D pix2pix U-Net (unet_128) | 66.6 M + D 11.1 M |
| `ct2pet` | 2.5D conditional GAN | Salehjahromi et al., *Cell Rep Med* 2024 (WuLabMDA/Synthetic-PET-from-CT) | ResUNet++ (7 slices → 3 slices) | 22.2 M + D 2.8 M |
| `jointsyn` / `jointsyn_paired` | unpaired cycle GAN + diagnosis | Wang et al., *Med Image Anal* 2024 (thibault-wch/Joint-Learning-for-Alzheimer-disease) | ShareSynNet (AdaIN U-Net) + 2 discriminators + AwareNet classifier | 33.0 M + 2×18.7 M + 16.0 M |

## Common protocol

* **Data**: `baselines/cohort.py` reuses TauGenNet's `TauPETDataset` unchanged — trilinear
  resample to 96×112×96, DK86 atlas mask, per-subject min-max to [0,1]; SUVR is recovered with
  the dataset's `get_pet_norms`. The manifest fixes the held-out test set and the pool (either
  `dev`, or `train`/`val` with a fixed validation list, optionally a `fold` column). CV folds are
  the manifest's folds or `StratifiedKFold(shuffle=True, random_state=42)` over the pool,
  stratified by diagnosis group — the mentor-split policy of `src/splits.py` (verified identical
  to TauGenNet's loader: same test order, same fold membership). Subjects without regional
  atrophy z-scores are excluded at manifest time, as TauGenNet's loader excludes them.
* **Model selection**: every trainer checkpoints the epoch with the lowest **validation** masked
  MSE of its MRI→PET prediction in [0,1] space; the held-out test set is never used for
  selection. Early stopping where the paper used it (patience in the table below).
* **Masking**: predictions are multiplied by the DK86 mask before scoring (as `generate.py`
  always did). The two regressors use the DK86-masked MSE of the Dense-U-Net baseline. The
  GAN pixel losses keep the papers' whole-volume means and weights (`--pixel-loss-masked`
  switches to in-mask means); Tanh models see data mapped to [-1,1] with background −1.
* **Scoring** (`scripts/score_baseline.py`, default `--pipeline github`): the lab's own
  evaluation pipeline from github.com/sophiazhou6/tau-pet-synthesis-baselines, checked out at
  `/scratch/network/sz3962/tau-pet-synthesis-baselines` (`TAU_GH_ROOT`) with its `taugennet/data/raw`
  symlinked to the raw data. Per run/fold: (1) `baselines/gh_align.py` rebuilds the held-out set
  with the GitHub loader and aborts unless it equals the manifest's test rows in the same order,
  and writes `per_subject_wholebrain.csv` (the `analysis/make_per_subject.py` convention);
  (2) `taugennet/scripts/evaluate_final.py --use-cached` is run exactly as
  `tau-denseunet/scripts/evaluate.py` runs it (DK86, SUVR, `--skip-ablation`, `--fold`,
  `--metrics-out`, glass brains) → `pooled_wholebrain_atrophy.csv` (SET1),
  `region_pooled_atrophy.csv` (SET2), `region_crosssubject_metrics_atrophy.csv` (SET3), the
  N×86 regional-mean CSVs, ROI tables, `metrics_atrophy.md`, `metrics.json` and all figures;
  (3) `taugennet/scripts/localization_metrics.py` → `localization.json` (SET3 / WITHIN / PEAK@10).
  `scripts/aggregate_baselines.py` builds the mean ± std table from these files with the
  RESULTS.md recipe (SET1 = the pooled CSV row, SET2/SET3 = mean of the `pearson` column).
  `--pipeline local` (the same metric functions imported from the local taugennet copy) exists
  only for debug runs with `--limit-subjects`.

## Model cards

### `denseunet` — 3D Dense-U-Net
Port of the Keras model of Lee et al. (`denseunet/model.py`, unchanged). Adam lr 1e-4, wd 0,
batch 2, masked MSE, ≤300 epochs, patience 40. This is the summer configuration; the grid-search
winner was lr 1e-4 / wd 0 (126 fixed epochs, `--epochs 126 --patience 0 --val-frac 0` reproduces
the "final" all-dev training).

### `swinunetr` — Swin UNETR
MONAI's `SwinUNETR` (the authors' implementation): patch 2, C = 48, depths [2,2,2,2], heads
[3,6,12,24], window 7, instance-norm residual decoder — paper Table 1. Changes for regression:
linear 1×1×1 output (no sigmoid), masked MSE instead of Dice; volumes padded from 112 → 128 in
the second axis (the network needs multiples of 32) and cropped back.
Optimisation: AdamW lr 1e-4, wd 1e-5, linear warm-up (10 %) + cosine, bf16 autocast, batch 2,
≤300 epochs, patience 40 — the official BraTS recipe of the authors' research-contributions
code (`optim_lr 1e-4, reg_weight 1e-5, warmup_cosine`). The paper text states lr 8e-4 for its
8-GPU run; run `--lr 8e-4` as a second configuration if the PI wants it selected by validation.
Augmentation (paper: flips on all axes, intensity shift ±0.1, scale 0.9–1.1): flips restricted
to left-right (the only anatomically valid flip for MNI-registered brains) + the intensity
jitter on the MRI; `--no-augment` disables.

### `eagan` / `deagan` — Ea-GANs
Official options (`train_gEaGAN.sh` / `train_dEaGAN.sh`): unet_128 generator (7 levels,
ngf 64, BatchNorm, dropout, Tanh), basic PatchGAN (ndf 64), LSGAN with label smoothing,
λ_L1 = 300, λ_Sobel = 100 ramped linearly over the first 20 epochs, Adam 2e-4 (β1 0.5),
100 epochs + 50 epochs linear decay, image pool 4, random flips. gEa-GAN puts the Sobel edge L1
in the generator; dEa-GAN also feeds the edge map to the discriminator.
Adaptations: volumes padded to 128³ (the paper's patch size; 7 levels need it) with background
−1; generator output masked to DK86; paper batch 6 → 2; flips only left-right; the official
dEa-GAN code also pushes *real* pairs through the fake-image pool (a bug) — not reproduced.

### `ct2pet` — 2.5D cGAN with ResUNet++
Configuration of the released checkpoint (`checkpoints/train_opt.txt`): netG ResUnetPlusPlus,
input 7 adjacent slices → output 3 central slices, ReLU output, BatchNorm; netD basic PatchGAN
on the 3 central source slices + 3 target slices; vanilla GAN; λ_L1 = 6000 (README: 4000);
Adam 2e-4 (β1 0.5); batch 5; 100 + 100 (linear decay) epochs; random affine augmentation
(p 0.95: rotation ±45°, translation 10 %, scale 0.85–1.15, shear ±7°) applied to source and
target; target intensities gamma-compressed (√) and squared back, clipped to [0,1].
Adaptations: axial (z) slices of the 96×112×96 volume, all 96 slabs of every subject each
epoch (the CT set was thinned to every 3rd slab); the volume is zero-padded by 3 slices so all
slices get a prediction; output masked to DK86. At inference every slab predicts its centre
slice (as in their `testNifty.py`).

### `jointsyn` / `jointsyn_paired` — joint synthesis + diagnosis
Re-implemented from the paper and the reference layer layout (the repository has no license):
* **ShareSynNet**: strided-conv encoder 1→32→128→256→512, attention (8 heads) bottleneck,
  transposed-conv decoder with skips, ConvT 32→4, 3×3×3 conv + Tanh; every stage
  Conv → InstanceNorm → AdaIN → LeakyReLU, AdaIN statistics from a learned input-independent
  code blended with weight α (α = 0: MRI→PET, α = 1: PET→MRI).
* **Discriminators** (`Defined`): 6 conv blocks (32,128,256,512,512,512), InstanceNorm, LSGAN.
* **AwareNet** classifier: 2D ResNet-18 stem on every axial slice, slice attention, two
  slice-shift + ResNet stages, GAP, per-slice classifier averaged over slices.
* **Training** (their `scripts/`): stage A ShareGAN pretraining — *unpaired* (MRI of subject i
  vs PET of a random subject j), cycle L1 λ 10, identity 0.3·λ, Adam 2e-4 β1 0.5, exponential
  decay 0.95^(epoch−15), ≤80 epochs, patience 20, gradient accumulation 2, pool 10; stage B
  AwareNet pretraining on real MRI, CE, Adam 1e-4, γ 0.95, batch 2, ≤80 epochs, patience 10;
  stage C joint fine-tuning, G lr 1e-6, classifier 1e-5, D 1e-6, λ_cls 1 with the CE term on the
  real MRI or the synthesized PET (50/50), lr × 0.7 per epoch, 15 epochs, batch 1.
* Their progressive 128³ → 256³ stages collapse into one stage at our resolution. Labels = the
  manifest diagnosis groups (AD vs MCI on the mentor cohort; CN/MCI/AD on a cohort with
  controls). `jointsyn` is the published, unpaired method; `jointsyn_paired` adds a paired L1
  term (λ 10, the cycle weight) in stages A and C because our data are paired. Diagnosis
  accuracy/AUC of the classifier on real MRI and on synthesized PET is logged per epoch
  (`history.json`) as a secondary result.

## Budget — measured on Adroit (fold 0 of the 235-subject mentor cohort, 150 train / 38 val, one A100 or A40)

| model | epochs run (max) | s/epoch | wall time incl. generate + score | GPU mem (batch 2) | checkpoint (best.pt) |
|---|---|---|---|---|---|
| denseunet | 184 (300, early-stopped) | 22 | 1 h 10 min | 7.2 GB | 171 MB |
| swinunetr | 145 (300, early-stopped) | 21 | 54 min | 11.8 GB | 244 MB |
| eagan | 150 | 7 | 22 min | 5.4 GB | 680 MB |
| deagan | 150 | 7 | 47 min* | 5.5 GB | 680 MB |
| ct2pet | 200 | 105 | ≈ 6 h | 2.9 GB | 95 MB |
| jointsyn(_paired) | 80 + 80 + 15 (early-stopped) | 25 (stage A) | ≈ 1.5 h | 5.3 GB | 329 MB + 329 MB (stage A) + 64 MB (stage B) |

\* shared a node with another job. Wall time under 4 h → the `gpu-short` QOS (4 GPUs at once);
`ct2pet` needs `--time=11:59:00` (`gpu-medium`, 2 GPUs; `23:59:00` on the larger cohort). Disk:
trainers keep only `best.pt` (add `--save-last` for a resume checkpoint); predictions are packed
losslessly into `predictions.npz` after scoring (11× smaller; `score_baseline.py` unpacks to
node-local `/tmp` when it needs them again; `--no-pack` keeps the `.npy` files). A full 7-model ×
5-fold cohort needs ≈ 17 GB of checkpoints + ≈ 2 GB of packed predictions.
`scripts/prune_checkpoints.py --cohort <name> --apply` removes leftovers of scored runs and packs
their predictions. The job script refuses to start above `QUOTA_MAX_GIB` (default 185) of the
200 GB scratch quota (`checkquota`). Time and storage for the expanded cohort: docs/PLAN.md.

## Open decisions for the PI

1. **Held-out test policy on the expanded cohort** — default of `prepare_cohort.py`: keep every
   previously held-out subject held out, keep previous dev subjects in dev, split new subjects
   80/20 stratified by group. Alternative: a fresh stratified split (`--test-frac` without
   `--test-from/--dev-from`).
2. **Joint framework**: report the published unpaired variant, the paired variant, or both.
3. **Diagnosis classes** for the joint framework on a CN+MCI+AD cohort (3-class vs CN-vs-AD).
4. **Swin UNETR learning rate**: 1e-4 (official code) vs 8e-4 (paper text) — both can be run
   and selected by validation.
5. **TauGenNet conditioning inputs** for the new subjects (regional atrophy z-scores, regional
   SUVR, plasma p-tau217) must be extended in the lab CSVs, otherwise those subjects are dropped
   by TauGenNet's loader (the baselines here need only MRI + PET).
