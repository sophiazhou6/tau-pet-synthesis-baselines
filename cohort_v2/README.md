# cohort_v2: the two baselines on the 597-subject cohort

Code used to run the 3D Dense-U-Net and TauGenNet on the expanded ADNI cohort (120 held-out test subjects,
477 development subjects). TauGenNet itself runs from `../taugennet/`; this folder holds the Dense-U-Net
port, the job scripts, and the scoring, figure and export steps. Paths are the ones used on Adroit.

No ADNI data, subject IDs, checkpoints or predictions are included.

## What is here

| Path | What it does |
|---|---|
| `baselines/`, `denseunet/` | Dense-U-Net port and training framework. `denseunet_paper` in `baselines/registry.py` is the configuration used. The other models in the registry were not part of this run. |
| `scripts/prepare_cohort.py`, `register_pinned_t1.py`, `qc_registration_full.py` | Build the cohort manifest, register the T1 scans, check the registrations. |
| `scripts/impute_plasma.py` | MICE imputation of missing plasma p-tau217, fit per split on training subjects only. |
| `scripts/train_baseline.py`, `generate_baseline.py`, `cv_round_schedule.py` | Dense-U-Net training, prediction, and the per-round median schedule for the final refit. |
| `scripts/score_baseline.py`, `baselines/gh_align.py` | Score predictions with `../taugennet/scripts/evaluate_final.py` and `localization_metrics.py`. |
| `scripts/cv_table.py`, `report_run.py`, `aggregate_baselines.py` | Metric tables with SDs. |
| `scripts/make_composite_figure.py`, `replot_pooled_scatter.py`, `plot_subject_figures.py` | Figure panels B and C and the A+B+C assembly. Panel A is `../analysis/render_glass_9case_white.py`. |
| `scripts/export_predictions_nifti.py` | Test-set NIfTI export. |
| `slurm/` | `baselines_cv` and `denseunet_refit` (Dense-U-Net folds and final model), `taugennet_paper` (TauGenNet folds and final model). |
| `audit/` | Checks described under "Reading the results". |

## How the runs were set up

Both models: 5-fold CV on the 477 development subjects, then a final model on all 477 scored once on the 120
test subjects. Stopping points for the final model come from the folds' medians.

- **Dense-U-Net:** MRI only, trained with the released recipe of Lee et al. 2024 (10 rounds of up to 15 epochs,
  learning rate halved between rounds, whole-volume MSE, Glorot initialization).
- **TauGenNet:** MRI plus plasma p-tau217, with missing plasma filled by MICE. Autoencoder with early stopping,
  then diffusion for a fixed 600 epochs, final checkpoint, 500-step sampling.

## Changes from the code in `../tau-denseunet/` and the earlier `RESULTS.md`

- New cohort, and MICE-imputed plasma.
- DK86 atlas placement fixed. Earlier results used a left-right mirrored atlas, so they are not directly comparable.
- Dense-U-Net uses the released training recipe instead of the earlier settings (up to 300 epochs, patience 40,
  masked MSE, constant learning rate).
- TauGenNet diffusion uses weight decay 0 and a corrected 500-step sampler.

## Reading the results

- **Scale.** Both models predict on a per-subject 0-1 scale, unlike the published Dense-U-Net, which predicts
  SUVR directly. The evaluator converts back to SUVR with each subject's own real PET min and max, so SET1, SET2,
  SET3, MSE and MAE depend on the real scan. Per-subject Pearson r does not.
  `audit/scale_leak.py` and `scale_leak2.py` re-score with one fixed scale and compare against the average
  training scan.
- **NIfTIs.** `export_predictions_nifti.py` defaults to one scale from the training subjects, so the files carry
  no real-scan information. These models were not trained to predict overall tau level, so the brain-wide level
  in those files is not a tau measurement: AD subjects come out dimmer than CN, the opposite of real scans.
  `audit/validate_fixed_nifti.py` checks the exported files.
- **Failed generation.** One TauGenNet fold's first run produced images that were not brains. It was repeated with
  a new noise seed (`TAUGENNET_RUN_SEED`, same subjects). `audit/shell_test.py` measures the outer shell of each
  predicted volume, which is near zero in a healthy prediction, and does not use any score.
- **CV vs test.** Test SET3 sits above the CV values because of the subjects, not the model.
  `audit/panelc.py` compares the fold models on validation and test subjects and places the test set among
  random groups drawn from the development pool.
