# Held-out TEST results — controls_322 cohort

All numbers below are recomputed directly from the CSVs committed under
`taugennet/results/records/` and `tau-denseunet/results/records/` — nothing here is transcribed
from a notebook or a log. To regenerate this table from the committed data:

- **SET1** = the single row of `pooled_wholebrain_atrophy.csv`
- **SET2** = mean of the `pearson` column of `region_pooled_atrophy.csv` (86 rows)
- **SET3** = mean of the `pearson` column of `region_crosssubject_metrics_atrophy.csv` (86 rows)

NRMSE / MSE / MAE / SSIM columns are the whole-brain pooled values (SET1 row). SSIM is
whole-brain only by design — see [README.md](README.md#the-three-metric-families).

Everything is in **SUVR (unnormalized)** space, DK86-masked, zero/out-of-brain voxels excluded.

> ⚠️ **Read the three caveats in [README.md](README.md#caveats-that-must-travel-with-these-numbers)
> before quoting any of these numbers.** In particular: the SUVR-conditioned arms are *not*
> MRI-only predictions, and the plasma arm has a different n.

## TauGenNet — 3D latent diffusion (DDPM, 500 steps)

Records: `taugennet/results/records/controls_322_TEST/<TAG>/`

| TAG | conditioning inputs | n | SET1 r | SET2 r | SET3 r | NRMSE | MSE | MAE | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| `final_t1_atrophy_enc` | MRI + atrophy | 64 | 0.4196 | 0.3564 | 0.4077 | 0.0956 | 0.4304 | 0.4754 | 0.8913 |
| `final_t2_suvr_enc` | MRI + SUVR | 64 | 0.5062 | 0.4483 | 0.5416 | 0.0848 | 0.3383 | 0.4180 | 0.9043 |
| `final_t2b_both_enc` | MRI + atrophy + SUVR | 64 | 0.5131 | 0.4464 | 0.5216 | 0.0770 | 0.2792 | 0.3789 | 0.9135 |
| `final_t3_suvr_enc_plasma` | MRI + SUVR + plasma p-tau217 | **48** | 0.5382 | 0.4516 | 0.5088 | 0.0732 | 0.2520 | 0.3607 | 0.9228 |

## DenseUNet — deterministic 3D regression

Records: `tau-denseunet/results/records/denseunet_baseline/<TAG>/`

| TAG | conditioning inputs | n | SET1 r | SET2 r | SET3 r | NRMSE | MSE | MAE | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| `controls322_plain_v2` | MRI only | 64 | 0.4891 | 0.4440 | 0.4665 | 0.0772 | 0.2808 | 0.3946 | 0.9175 |
| `controls322_paint_atrophy` | MRI + atrophy | 64 | 0.5194 | 0.4344 | 0.4484 | 0.0640 | 0.1930 | 0.3370 | 0.9260 |
| `controls322_suvr_lr1e-4_wd0` | MRI + SUVR (painted) | 64 | 0.5954 | 0.5110 | 0.4860 | 0.0564 | 0.1496 | 0.2766 | 0.9469 |
| `controls322_paint_both` | MRI + atrophy + SUVR | 64 | 0.5761 | 0.4910 | 0.4716 | 0.0598 | 0.1684 | 0.2992 | 0.9420 |

A fifth DenseUNet variant (`controls322_suvrfilm_lr1e-4_wd0`, SUVR delivered as a FiLM vector
rather than a painted channel) was run but is deliberately excluded from this comparison; its
records are not committed here.

## Comparability

All seven n=64 arms above share an **identical TEST subject set and identical row order** —
the two models were evaluated on literally the same 64 subjects, in the same order, through the
same `evaluate_final.py` metric code. The n=48 plasma arm is a strict subset of that same set.

The two atrophy arms (`final_t1_atrophy_enc` and `controls322_paint_atrophy`) receive
*identical conditioning content*, not merely equivalent inputs: DenseUNet has no dataset layer
of its own, it imports TauGenNet's `src.dataset_final` and reads the same `TAUGENNET_PAINT`
environment variable, so both read the same 86-dim regional atrophy z-score file through the
same code path. The two models differ only in how each consumes that painted map — DenseUNet
concatenates it as a full-resolution input channel, TauGenNet routes it through
`AtrophyEncoder3D` down to latent resolution.

## Cross-validation coverage

Held-out TEST is a one-time final evaluation. Where 5-fold CV (scored on the *validation* split)
exists, it is under `records/controls_322_cv/<ARM>/fold_{0..4}/` on the cluster/archive — those
fold-level records are **not** committed here (see the curation note in
[README.md](README.md#what-is-and-isnt-committed)).

| TauGenNet TEST config | 5-fold CV arm | DenseUNet TEST config | 5-fold CV arm |
|---|---|---|---|
| `final_t1_atrophy_enc` | `cv_atrophymap_learnedenc` | `controls322_plain_v2` | `cv322_grid322_lr1e-4_wd0.0` |
| `final_t2_suvr_enc` | `cv_suvrmap_learnedenc` | `controls322_paint_atrophy` | *(none — TEST only)* |
| `final_t2b_both_enc` | `cv_bothmaps_enc` | `controls322_suvr_lr1e-4_wd0` | *(none — TEST only)* |
| `final_t3_suvr_enc_plasma` | `cv_suvrenc_plasma` | `controls322_paint_both` | *(none — TEST only)* |

Do not substitute the `cv_atrophy*` DenseUNet arms for the three missing ones — those are a
legacy AD/MCI cohort (~38 subjects/fold vs. controls_322's ~51), a different population.

## Plasma p-tau217 ablation — null result

`final_t3` (n=48) vs. `final_t2` (n=64) is a cross-cohort comparison, **not** a plasma ablation.
The valid ablation is the pair of matched-cohort 5-fold validation arms:

| arm (val, 5-fold, matched cohort) | SET1 | SET2 | SET3 |
|---|---|---|---|
| `cv_suvrenc_noplasma_matched` | 0.4861 ± 0.0552 | 0.4217 ± 0.0581 | 0.4945 ± 0.0916 |
| `cv_suvrenc_plasma` | 0.4822 ± 0.0551 | 0.4187 ± 0.0551 | 0.4953 ± 0.0820 |
| Δ (plasma − no-plasma) | −0.0039 | −0.0030 | +0.0008 |

Every delta is under 0.004 against a fold SD of 0.055–0.092, and inconsistent in sign. Plasma
p-tau217 conditioning provides no measurable benefit in this setup. This is a reportable null,
not a missing experiment.
