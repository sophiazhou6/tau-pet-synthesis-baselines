# tau-pet-synthesis-baselines

Two published approaches to MRI→tau-PET synthesis, reimplemented in PyTorch and evaluated
head-to-head on the same ADNI cohort, the same split, and the same metric code.

| Folder | Model |
|---|---|
| [`taugennet/`](taugennet/) | 3D **latent diffusion** — MRI-conditioned DDPM, 500 sampling steps, volumes 96×112×96. After Gong et al., ["TauGenNet: Plasma-Driven Tau PET Image Synthesis via Text-Guided 3D Diffusion Models"](https://arxiv.org/abs/2509.04269), IEEE TRPMS. |
| [`tau-denseunet/`](tau-denseunet/) | Deterministic 3D **Dense-U-Net** — direct MRI→tau-PET regression. Port of [Neurology-AI-Program/AI_imputed_tau_PET](https://github.com/Neurology-AI-Program/AI_imputed_tau_PET), the code accompanying Lee et al., ["Synthesizing images of tau pathology from cross-modal neuroimaging using deep learning"](https://pubmed.ncbi.nlm.nih.gov/37804318/), *Brain* 147(3):980–995, 2024. |

The two share a data layer: DenseUNet imports TauGenNet's `src.dataset_final` rather than
defining its own, so both models see identical inputs and are scored by identical metric code.

Results tables: **[RESULTS.md](RESULTS.md)**.

## Setup

```bash
export TAUGENNET_ROOT=/path/to/this/repo/taugennet
export PYTHONPATH=$TAUGENNET_ROOT:${PYTHONPATH:-}
```

`TAUGENNET_ROOT` is read by `taugennet/src/config.py` and by DenseUNet to locate the shared
dataset layer — set it before running anything.

Python 3.9. Requires `torch`, `nibabel`, `numpy`, `scipy`, `scikit-image`, `nilearn`,
`matplotlib`, `pandas`, `tqdm`, and `transformers` (the last only for the `ptau217` /
`combined` conditioning modes, which use a frozen text encoder).

No ADNI data or trained checkpoints are included. Both models expect a locally populated
`data/raw/`; `taugennet/src/config.py` and `taugennet/src/dataset_final.py` document the
expected layout.

---

# `taugennet/` — diffusion model

## `src/` — the model

| File | What it is |
|---|---|
| `models.py` | Base 3D UNet + autoencoder. |
| `models_spatial.py` | UNet variant taking a spatial conditioning map, plus `AtrophyEncoder3D`, which downsamples a full-resolution painted map to latent resolution. |
| `diffusion.py` | Linear beta schedule, forward diffusion, DDPM reverse step. |
| `conditioning.py` | Interchangeable conditioning strategies — atrophy MLP, frozen text encoder for plasma p-tau217, learned-null context. |
| `dataset_final.py` | Main ADNI loader. Shared with DenseUNet. |
| `dataset_spatial.py` | Loader for spatial-map conditioning; paints regional values into a volume via the DK86 atlas. |
| `dataset_combined.py` | Loader for combined atrophy + p-tau217 conditioning. |
| `atlas_labels.py` | DK86 atlas label → region-column mapping. |
| `splits.py` | Subject splits for 5-fold CV + held-out test. |
| `ema.py` | Exponential moving average of model weights. |
| `inference.py` | Sampling loop. |
| `config.py` | Paths, volume shape, seed. |

## `scripts/` — training

| Script | What it does |
|---|---|
| `train.py` | Trains the autoencoder and diffusion model. `--mode` picks conditioning: `atrophy`, `ptau217`, or `combined`. |
| `train_spatial_atrophy.py` | Trains with an atlas-painted spatial conditioning map. `--cond-mode` selects the cross-attention conditioner alongside it; `--use-spade` swaps in SPADE normalization. |
| `train_cfg.py` | Diffusion training with classifier-free guidance. |
| `train_ae.py` | Autoencoder only. |
| `train_lp_modality.py` | Adds optional 3D LP and/or modality-aware autoencoder loss. |
| `train_dynamic_prompt.py` | CoMA-style dynamic prompt head variant. |
| `run_cv.py` | Orchestrates 5-fold CV training. |
| `run_grid_search.py` | Three-stage coordinate-descent hyperparameter search. |
| `run_experiment.py` | End-to-end: submit, poll, evaluate, write tables. |

## `scripts/` — generation and evaluation

| Script | What it does |
|---|---|
| `generate_spatial.py` | Runs DDPM sampling for the spatial-conditioning model and caches volumes as `.npy`. |
| `generate_posterior_mean.py` | Averages K DDPM samples per subject into a posterior mean. |
| `evaluate_final.py` | **The canonical evaluator.** Computes all three metric families (below) in SUVR space, DK86-masked, writes the CSVs, and renders glass brains. Used for both models. |
| `evaluate_cfg.py` | Evaluator for classifier-free-guidance checkpoints. |
| `evaluate_tissue.py` | Tissue-stratified evaluation. |
| `eval_ae_recon.py` | Autoencoder reconstruction quality alone, isolating AE error from diffusion error. |
| `crosssubject_r.py` | Voxelwise cross-subject Pearson r from cached generations. |
| `peak_from_gens.py` | Computes cross-subject / within-subject / peak metrics straight from cached `.npy` files. |
| `localization_metrics.py` | Ranks runs by *where* tau is placed rather than overall burden. |

## `scripts/` — figures and aggregation

| Script | What it does |
|---|---|
| `glass_brain_final.py` | Glass-brain, orthogonal-slice, and scatter renders. `rank_by` selects best/median/worst exemplar subjects by a chosen metric. |
| `suvr_eval_figures.py` | SET1 hexbin, SET2/SET3 region bars, SET2-vs-SET3 scatter. |
| `paper_figures.py`, `recreate_paper_figures.py` | Publication figure sets from cached results. |
| `scatter_plot.py` | Generated-vs-real scatter per tau-relevant ROI. |
| `roi_table.py`, `metrics_table.py` | Metric tables from records / SLURM logs. |
| `aggregate_cv_eval.py` | Per-fold metrics → CV mean ± std. |
| `aggregate_cond_compare.py` | Conditioning-variant CV runs → one comparison table. |
| `model_comparison.py`, `make_verification_figure.py` | Architecture-variant comparison figures. |

## `slurm/`

Cluster job scripts, grouped by purpose: `train/`, `eval/`, `cv/`, `grid_search/`,
`arch_sweep/`, `controls_322/`, `ablation/`. Each wraps one of the scripts above with the
resource request and environment setup.

---

# `tau-denseunet/` — Dense-U-Net baseline

| File | What it does |
|---|---|
| `denseunet/model.py` | `DenseUNet3D` — dense-block 3D U-Net. `in_ch` grows when conditioning maps are concatenated; `film_cond_dim > 0` enables FiLM modulation from a conditioning vector. |
| `denseunet/config.py` | Paths and hyperparameters; re-exports `build_dataloaders` from TauGenNet's dataset layer. |
| `denseunet/masks.py` | Builds and caches the whole-brain mask, injects it into datasets. |
| `denseunet/figs.py` | Scatter and region-bar plots. |
| `scripts/train.py` | Trains the model (MRI → tau PET regression). Adam, or AdamW when `--weight-decay > 0`. |
| `scripts/generate.py` | Predicts the test set and caches volumes in the same `.npy` layout TauGenNet's evaluator reads. |
| `scripts/evaluate.py` | Scores predictions with TauGenNet's canonical metrics. |
| `scripts/suvr_sets.py` | Scores one fold, emitting whole-brain per-subject metrics and the SET1/2/3 CSVs. |
| `scripts/aggregate_cv.py`, `aggregate_suvr_sets.py` | Roll per-fold results into CV summaries. |
| `scripts/grid_summary.py` | Ranks grid configs by mean validation loss across folds. |
| `scripts/cross_eval.py` | Scores a whole-brain-trained model over the DK86 region mask. |
| `scripts/glass_perfold.py`, `peak_and_folds.py` | Per-fold glass brains and per-fold metric products. |
| `docs/` | The upstream Keras reference implementation, kept for comparison. |

See [`tau-denseunet/README.md`](tau-denseunet/README.md) for what the port changes relative to
upstream, and [`tau-denseunet/GRID_SEARCH_RUNBOOK.md`](tau-denseunet/GRID_SEARCH_RUNBOOK.md) for
the hyperparameter search procedure.

---

# `analysis/`

Figure and rescoring scripts used to produce the final comparison outputs.

| Script | What it does |
|---|---|
| `render_8rid_all.py` | Slice panels for 8 subjects across every model variant, on a shared intensity scale. |
| `render_9case_slices.py` | 9-subject slice grid with a custom tau colormap. |
| `render_glass_9case.py`, `_white.py`, `_cached.py` | 9-subject glass-brain sheets — default, white-background, and cached-generation variants. |
| `render_glass_2row.py` | Two-row glass-brain comparison sheet. |
| `make_per_subject.py`, `make_per_subject_taugennet.py` | Per-subject metric CSVs for each model. |
| `rescore_cv_persubject.py` | Per-subject metrics from cached CV generations, given an arm and conditioning mode. |
| `pick_winner.py` | Ranks grid configs by per-fold validation loss read from the checkpoints. |
| `extract_fills.py` | Pulls metric values out of the records CSVs for table population. |

---

# Evaluation outputs

`evaluate_final.py` scores both models identically, in SUVR (unnormalized) space, masked to the
86 Desikan–Killiany regions, and writes three families of CSV:

| | Family | Computation | CSV |
|---|---|---|---|
| **SET1** | Pooled voxel-level, whole brain | All in-brain voxels of all subjects pooled into one real and one generated vector → single Pearson r / NRMSE / MSE / MAE, plus per-subject-averaged SSIM | `pooled_wholebrain_{mode}.csv` |
| **SET2** | Pooled voxel-level, per region | Same pooling, restricted to each of the 86 regions | `region_pooled_{mode}.csv` |
| **SET3** | Regional cross-subject | Average each region's voxels per subject (N×86), then correlate real vs. generated regional means across subjects | `region_crosssubject_metrics_{mode}.csv` |

SET1 and SET2 measure voxel-level image fidelity; SET3 measures how well the model separates
subjects. SSIM is reported whole-brain only, since a windowed metric evaluated inside a single
region's bounding box picks up neighbouring tissue.

The CSVs behind [RESULTS.md](RESULTS.md) are committed under each model's `results/records/`.
Checkpoints, generated volumes, figures, and ADNI data are not committed.

## Quickstart

```bash
# Diffusion model
cd taugennet
python scripts/train.py --mode atrophy
python scripts/evaluate_final.py --mode atrophy \
    --checkpoint-dir results/checkpoints/<run> \
    --save-generated --generated-dir results/generated/<variant>/atrophy

# Dense-U-Net
cd tau-denseunet
python scripts/train.py    --mode atrophy
python scripts/generate.py --mode atrophy

# Score either one with the same evaluator
cd $TAUGENNET_ROOT
python scripts/evaluate_final.py --mode atrophy --use-cached \
    --generated-dir <path-to-generations>
```

Give `--generated-dir` a distinct path per variant so runs don't overwrite each other.
`--no-glass-brain` skips figure rendering.
