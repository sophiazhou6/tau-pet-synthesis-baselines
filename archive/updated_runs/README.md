# updated_runs: the two baselines on the 597-subject cohort

Code for the November 2026 runs of the 3D Dense-U-Net and TauGenNet on the expanded ADNI cohort (120 held-out
test subjects, 477 development subjects). Nothing outside this folder was changed. The summer code and
`RESULTS.md` are tagged `summer-2026`.

No ADNI data, subject IDs, checkpoints or predictions are included. Paths are the ones used on Adroit.

## What is here

| Path | What it does |
|---|---|
| `taugennet_changes/summer-2026_to_updated_runs.patch` | The changes these runs made to `../taugennet/` and `../analysis/render_glass_9case_white.py` (10 files). Apply from the repository root with `git apply updated_runs/taugennet_changes/summer-2026_to_updated_runs.patch` before running TauGenNet. |
| `baselines/`, `denseunet/` | Dense-U-Net port and training framework. `denseunet_paper` in `baselines/registry.py` is the configuration used. The other models in the registry were not part of these runs. |
| `scripts/prepare_cohort.py`, `register_pinned_t1.py`, `qc_registration_full.py` | Build the cohort manifest, register the T1 scans, check the registrations. |
| `scripts/impute_plasma.py` | MICE imputation of missing plasma p-tau217, fit per split on training subjects only. |
| `scripts/train_baseline.py`, `generate_baseline.py`, `cv_round_schedule.py` | Dense-U-Net training, prediction, and the per-round median schedule for the final model. |
| `scripts/score_baseline.py`, `baselines/gh_align.py` | Score predictions with `../taugennet/scripts/evaluate_final.py` and `localization_metrics.py`. |
| `scripts/cv_table.py`, `report_run.py`, `aggregate_baselines.py` | Metric tables with SDs. |
| `scripts/make_composite_figure.py`, `replot_pooled_scatter.py`, `plot_subject_figures.py` | Figure panels B and C and the A+B+C assembly. Panel A is `../analysis/render_glass_9case_white.py` after the patch. |
| `scripts/render_glass_all_subjects.py` | Glass-brain figures for every test subject, ground truth and both models. |
| `scripts/export_predictions_nifti.py` | Test-set NIfTI export. |
| `slurm/` | `baselines_cv` and `denseunet_refit` (Dense-U-Net folds and final model), `taugennet_paper` (TauGenNet folds and final model). |
| `audit/` | Checks described under "Reading the results". |

## What the patch changes in TauGenNet

- The controls_322 route accepts plasma p-tau217 (CN subjects included) and reads the per-split MICE table.
- The DK86 atlas is placed through the image headers. The summer loaders resized the raw atlas array onto the
  scans, which mirrored it left to right and shrank it about 6%. This changes every masked and regional metric.
- `--diff-weight-decay` (0 for these runs, as in Gong et al.), a corrected 500-step DDPM sampler, training without
  a validation set for the final model, and a save at the last epoch.
- `TAUGENNET_RUN_SEED` changes only training and sampling noise. `SEED` also sets the CV folds, so it stays fixed.
- `render_glass_9case_white.py` uses hot_r, the lyrz views and one Tau SUVR colour bar, and reads subject IDs
  from `GLASS_RIDS` instead of the source.

## How the runs were set up

Both models: 5-fold CV on the 477 development subjects, then a final model on all 477 scored once on the 120
test subjects. Stopping points for the final model come from the folds' medians.

- **Dense-U-Net:** MRI only, trained with the released recipe of Lee et al. 2024 (10 rounds of up to 15 epochs,
  learning rate halved between rounds, whole-volume MSE, Glorot initialization). The summer runs used up to 300
  epochs, patience 40, masked MSE and a constant learning rate.
- **TauGenNet:** MRI plus plasma p-tau217 through frozen CLIP ("Plasma is X.XXX."), with missing plasma filled
  by MICE. Autoencoder with early stopping, then diffusion for a fixed 600 epochs, final checkpoint, 500-step
  sampling.

## Reading the results

- **Scale.** Both models predict on a per-subject 0-1 scale, unlike the published Dense-U-Net, which predicts
  SUVR directly. The evaluator converts back to SUVR with each subject's own real PET min and max, so SET1, SET2,
  SET3, MSE and MAE depend on the real scan. Per-subject Pearson r does not. `audit/scale_leak.py` and
  `scale_leak2.py` re-score with one fixed scale and compare against the average training scan.
- **Voxel-level vs subject-level.** Every subject is scored on the same DK86 mask, so voxel-level and
  subject-level MSE and MAE are the same number by construction. RMSE and Pearson r differ.
- **NIfTIs.** `export_predictions_nifti.py` defaults to one scale from the training subjects, so the files carry
  no real-scan information. These models were not trained to predict overall tau level, so the brain-wide level
  in those files is not a tau measurement. `audit/validate_fixed_nifti.py` checks the exported files.
- **Failed generation.** One TauGenNet fold's first run produced images that were not brains. It was repeated with
  a new noise seed on the same subjects. `audit/shell_test.py` measures the outer shell of each predicted volume,
  which is near zero in a healthy prediction, and does not use any score.
- **CV vs test.** Test SET3 sits above the CV values because of the subjects, not the model.
  `audit/panelc.py` compares the fold models on validation and test subjects and places the test set among
  random groups drawn from the development pool.
