# Final code: the two baselines on the 597-subject cohort (November 2026)

Only the files the final runs used. Each is unchanged except `baselines/registry.py`, which no longer lists the
GAN baselines (see below). Paths point to the Adroit cluster. No ADNI data, subject IDs, checkpoints or
predictions are included.

## Where things are

| | Dense-U-Net (MRI only) | TauGenNet (MRI + plasma p-tau217) |
|---|---|---|
| Model | `denseunet_and_pipeline/denseunet/model.py` | `taugennet/src/models.py` (autoencoder, denoising U-Net), `src/conditioning.py` (`PTau217Conditioner`: frozen CLIP, prompt "Plasma is X.XXX.") |
| Training | `scripts/train_baseline.py --model denseunet_paper` (round schedule in `baselines/trainers/regression.py`) | `taugennet/scripts/train.py --mode ptau217` (autoencoder first, then diffusion) |
| Prediction | `scripts/generate_baseline.py` | `taugennet/scripts/evaluate_final.py --mode ptau217 --n-steps 500` (sampler in `src/inference.py`, `src/diffusion.py`) |
| Job script | `slurm/baselines_cv.slurm` (5 folds), `slurm/denseunet_refit.slurm` (final model) | `slurm/taugennet_paper.slurm` (folds and final model) |

Used by both models (paths not starting with `taugennet/` are in `denseunet_and_pipeline/`):
- Data loading: `taugennet/src/dataset_final.py`, with the DK86 atlas from `taugennet/src/atlas_labels.py`.
- CV folds: `taugennet/src/splits.py` for TauGenNet and `baselines/cohort.py` for Dense-U-Net. Both use the same rule and seed (42), so the folds hold the same subjects.
- Plasma: `scripts/impute_plasma.py`. MICE is fitted per split on that split's training subjects only.
- Scoring: `scripts/score_baseline.py`, which calls `taugennet/scripts/evaluate_final.py`.
- Tables: `scripts/cv_table.py` and `scripts/report_run.py`.
- NIfTI export: `scripts/export_predictions_nifti.py`.
- Cohort setup and T1 registration, run once: `scripts/prepare_cohort.py`, `register_pinned_t1.py` and `qc_registration_full.py`.

## Order of a run

1. `impute_plasma.py` writes one plasma table per fold, plus one for the final model.
2. Dense-U-Net: run `baselines_cv.slurm` for each fold, then `denseunet_refit.slurm`. The final model trains on all
   477 development subjects, using the folds' median stopping point for each round.
3. TauGenNet: run `taugennet_paper.slurm`. For each fold and for the final model it trains the autoencoder (early
   stopping), then trains diffusion for a fixed 600 epochs, then samples and scores the subjects. Fold models do this for their
   validation and test subjects, the final model for the test subjects.

## Things to know

- `baselines/registry.py` lost three imports and the entries for the GAN baselines (Ea-GAN, CT2PET, JointSyn),
  whose files are not included. The `denseunet_paper` settings are unchanged.
- Two files are here only because the code imports them at start-up; these runs did not use them:
  `taugennet/src/dataset_combined.py` and `baselines/models/swin_unetr.py`.
- TauGenNet's code still has options for its summer variants (ReLU, spatial conditioning, phase 6), but those
  files are left out. Only the setting used here runs: `--mode ptau217 --arch silu --dataset final`.
- The Dense-U-Net jobs loaded data with an older copy of `TauPETDataset` (`taugennet/src/dataset_final.py`). The
  version here only adds an option that is off by default, so it loads the same data.
- TauGenNet needs the CLIP text encoder `openai/clip-vit-base-patch32` in `taugennet/clip_model/`. It is not
  included; download it from Hugging Face.
- Both models predict each subject on a 0-1 scale.
  - The scorer converts predictions back to SUVR with each subject's own real PET min and max.
  - `export_predictions_nifti.py` instead uses one fixed scale taken from the training subjects. It is the default,
    and the shared NIfTIs were made this way.
- The figure scripts and the checks behind the results are in `../archive/updated_runs/` (`scripts/`, `audit/`).
  The summer code is in `../archive/summer-2026/`.
