# Results — held-out test set

Both models, same cohort, same held-out test subjects, same evaluator
(`taugennet/scripts/evaluate_final.py`). All values in SUVR space, DK86-masked.

- **SET1** — pooled voxel-level Pearson r, whole brain
- **SET2** — pooled voxel-level Pearson r, averaged over the 86 DK regions
- **SET3** — regional cross-subject Pearson r, averaged over the 86 DK regions

NRMSE / MSE / MAE / SSIM are the whole-brain pooled values. Every number below is computed from
the CSVs committed under each model's `results/records/`.

## TauGenNet — 3D latent diffusion (DDPM, 500 steps)

`taugennet/results/records/controls_322_TEST/<TAG>/`

| TAG | conditioning inputs | n | SET1 | SET2 | SET3 | NRMSE | MSE | MAE | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| `final_t1_atrophy_enc` | MRI + atrophy | 64 | 0.4196 | 0.3564 | 0.4077 | 0.0956 | 0.4304 | 0.4754 | 0.8913 |
| `final_t2_suvr_enc` | MRI + SUVR | 64 | 0.5062 | 0.4483 | 0.5416 | 0.0848 | 0.3383 | 0.4180 | 0.9043 |
| `final_t2b_both_enc` | MRI + atrophy + SUVR | 64 | 0.5131 | 0.4464 | 0.5216 | 0.0770 | 0.2792 | 0.3789 | 0.9135 |
| `final_t3_suvr_enc_plasma` | MRI + SUVR + plasma p-tau217 | 48 | 0.5382 | 0.4516 | 0.5088 | 0.0732 | 0.2520 | 0.3607 | 0.9228 |

## Dense-U-Net — deterministic regression

`tau-denseunet/results/records/denseunet_baseline/<TAG>/`

| TAG | conditioning inputs | n | SET1 | SET2 | SET3 | NRMSE | MSE | MAE | SSIM |
|---|---|---|---|---|---|---|---|---|---|
| `controls322_plain_v2` | MRI only | 64 | 0.4891 | 0.4440 | 0.4665 | 0.0772 | 0.2808 | 0.3946 | 0.9175 |
| `controls322_paint_atrophy` | MRI + atrophy | 64 | 0.5194 | 0.4344 | 0.4484 | 0.0640 | 0.1930 | 0.3370 | 0.9260 |
| `controls322_suvr_lr1e-4_wd0` | MRI + SUVR | 64 | 0.5954 | 0.5110 | 0.4860 | 0.0564 | 0.1496 | 0.2766 | 0.9469 |
| `controls322_paint_both` | MRI + atrophy + SUVR | 64 | 0.5761 | 0.4910 | 0.4716 | 0.0598 | 0.1684 | 0.2992 | 0.9420 |

## Reproducing these tables

```bash
# SET1 — the single row of the pooled CSV
cat taugennet/results/records/controls_322_TEST/final_t2_suvr_enc/pooled_wholebrain_atrophy.csv

# SET2 / SET3 — mean of the pearson column across the 86 regions
python -c "import pandas as pd; \
print(pd.read_csv('taugennet/results/records/controls_322_TEST/final_t2_suvr_enc/region_pooled_atrophy.csv').pearson.mean())"
```
