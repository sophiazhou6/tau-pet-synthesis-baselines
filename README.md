# tau-pet-synthesis-baselines

Two published approaches to MRI→tau-PET synthesis, reimplemented and evaluated head-to-head on
the same ADNI cohort, the same held-out split, and the same metric code. This is a
baseline-comparison study — a reference point for other tau-PET-synthesis work, not a novel
model in its own right.

**Headline results: [RESULTS.md](RESULTS.md).**

## The two models

| Folder | What it is |
|---|---|
| [`taugennet/`](taugennet/) | A 3D **latent diffusion** model — MRI-conditioned DDPM, **500 sampling steps** (not DDIM), input volumes **96×112×96**. Follows Gong et al., ["TauGenNet: Plasma-Driven Tau PET Image Synthesis via Text-Guided 3D Diffusion Models"](https://arxiv.org/abs/2509.04269) (IEEE TRPMS), extended here with a spatial-map conditioning path, a learned atrophy encoder, and an architecture sweep. |
| [`tau-denseunet/`](tau-denseunet/) | A **deterministic 3D Dense-U-Net** doing direct MRI→tau-PET regression — a PyTorch port of [Neurology-AI-Program/AI_imputed_tau_PET](https://github.com/Neurology-AI-Program/AI_imputed_tau_PET) (Kolařík et al. 2019). |

**The two models are coupled, deliberately.** DenseUNet has no dataset layer of its own: it
imports TauGenNet's `src.dataset_final` via `sys.path` and reads the same `TAUGENNET_PAINT`
environment variable. That is why they live in one repository — it is what guarantees both
models see byte-identical conditioning inputs rather than "comparable" ones. Setting
`TAUGENNET_ROOT` is a hard prerequisite for running *either* model.

## Cohort and split

The cohort used for every number in [RESULTS.md](RESULTS.md) is **`controls_322`**: an ADNI
tau-PET + MRI cohort including cognitively normal controls, split into a development set and a
one-time held-out TEST set.

- **321 subjects in the split files → 317 actually loaded** (see defect 1 below)
- **253 development / 64 held-out TEST**
- Volumes are **96×112×96**
- Diagnostic composition of the n=64 TEST set: MCI 31, CN 18, AD 15
- The n=48 plasma subset: MCI 22, CN 16, AD 10

Held-out TEST is spent exactly once, at the end. Everything used for model selection —
hyperparameter grids, architecture sweeps, conditioning comparisons — is scored 5-fold on the
**validation** split, never on TEST.

Provenance note: the derivation of the regional atrophy z-scores, and of the CN/MCI/AD group
labels, is **not traced** in this repository. Both were read from CSVs assembled upstream. No
claim is made here about how either was computed.

## The three metric families

Both models are scored by one shared script — `taugennet/scripts/evaluate_final.py`. No metric
code is duplicated between the two model directories. All three families are computed in
**SUVR (unnormalized) space**, **DK86-masked** (Desikan–Killiany, 86 regions, nearest-neighbour
interpolated to volume shape), with zero / out-of-brain voxels excluded.

| | Family | What it measures | Output CSV |
|---|---|---|---|
| **SET1** | Pooled voxel-level, whole brain | All in-brain voxels of all subjects pooled into one real vector and one generated vector → a single Pearson r / NRMSE / MSE / MAE, plus per-subject-averaged SSIM | `pooled_wholebrain_{mode}.csv` |
| **SET2** | Pooled voxel-level, per region | Same pooling, restricted to each of the 86 DK regions | `region_pooled_{mode}.csv` (86 rows) |
| **SET3** | Regional cross-subject | Average each region's voxels *per subject* (N×86 matrix), then correlate real vs. generated regional means *across subjects* | `region_crosssubject_metrics_{mode}.csv` (86 rows) |

SET1 and SET2 measure voxel-level image fidelity. SET3 measures whether the model separates
*patients* correctly — a model can score well on SET1 by reproducing anatomy and still have no
SET3 signal.

**SSIM is reported at whole-brain level only, by design.** It is a windowed, spatial metric, so
evaluating it inside a single region's bounding box leaks intensity from neighbouring tissue.
The 86-region tables therefore carry Pearson / NRMSE / MSE / MAE and omit SSIM. This is not an
oversight.

**SUVR reconstruction:** unnormalization uses the dataset's own `get_pet_norms` (masked
min/max). Never fall back to raw-NIfTI min/max — the raw max includes out-of-brain hot voxels
(~20 vs. a true in-brain ~5–7) and inflates SET1/SET3 by roughly 1.5–2.4×.

## Caveats that must travel with these numbers

1. **The SUVR-conditioned arms are not MRI-only predictions.** The regional SUVR conditioning
   input correlates **r = 0.870** (median, per-region cross-subject) with the target's own
   regional means. That is substantial leakage — those arms are partly reconstructing an input.
   The **atrophy-only arms** (`final_t1_atrophy_enc`; `controls322_plain_v2`,
   `controls322_paint_atrophy`) are the honest MRI-only comparison. Any figure, caption, or
   table using a SUVR-conditioned arm must say this.

2. **The plasma arm is n=48, the rest are n=64.** Never plot `final_t3_suvr_enc_plasma` on a
   shared axis with the n=64 arms without labelling the different n, and never read
   `final_t3` vs. `final_t2` as a plasma ablation — that is a cross-cohort comparison. The
   real ablation (matched cohort, 5-fold validation) is in
   [RESULTS.md](RESULTS.md#plasma-p-tau217-ablation--null-result) and is a **null result**.

3. **SUVR conditioning cuts MSE by ~50% but moves Pearson only ~4–20%.** It supplies intensity
   calibration, not spatial pattern. A Pearson-only table hides this — always show the full
   metric set (Pearson / NRMSE / MSE / MAE, plus whole-brain SSIM).

## Known defects — documented, not hidden

### 1. RID zero-padding bug — **present in this code, unfixed**

`src/dataset_spatial.py` and `src/dataset_final.py` parse ADNI RIDs two different ways and then
compare them as strings:

```python
# MRI directories look like "006_S_0731"
rid = os.path.basename(subj_dir).split("_")[-1]        # -> "0731"
#   dataset_spatial.py:236, dataset_final.py:194

# PET directories look like "RID_731"
rid = os.path.basename(subj_dir).replace("RID_", "")   # -> "731"
#   dataset_spatial.py:385, dataset_final.py:288 and :440
```

`"0731" != "731"`, so the membership test fails and the subject is dropped **with no warning**.
Only RIDs below 1000 are affected — those are the only ones zero-padded to four digits in the
`<site>_S_<rid>` MRI naming convention.

**Measured impact: 321 split subjects → 317 loaded. TEST 65 → 64 (dropping RID 731, a CN
subject); dev 256 → 253.** Every number in this repository is an n=317 number.

The fix is a `_norm_rid(raw)` helper returning `str(int(raw)) if raw.isdigit() else raw`,
wrapping **both** parse sites in **both** files. It has been drafted but deliberately **not
applied**: applying it would invalidate every cached generation, records CSV, and figure behind
the current results. If you patch it, patch both files — patching only one makes the two models
silently run on different cohorts (321 vs. 317), which is worse than the bug. After patching, a
fresh run yields 321 subjects (65 TEST / 256 dev); never mix pre- and post-patch numbers in one
table. The loader's `Matched subjects (...): N` print is the tell — 317 means the old path, 321
means patched.

Anyone cloning this repository and running it as-is reproduces n=317.

### 2. `ptau217_mlp` conditioning mode is untested

`scripts/train_spatial_atrophy.py` implements a `ptau217_mlp` conditioning branch in the
function body (around line 566) while the `--cond-mode` argparse `choices` list omitted it. Any
run requesting that mode exited in about 5 seconds with a **clean exit code 0:0**, and the
chained evaluation then silently skipped every fold — a false-clean failure, not a crash. This
produced one entirely empty CV arm (`cv_suvrenc_plasmamlp`, 0 scored folds); do not use it for
anything. The argparse list was patched on 2026-08-06 but **no successful run has completed
since**, so this mode remains unexercised.

### 3. Weight decay was a silent no-op in part of the DenseUNet grid

In the original 6-cell DenseUNet hyperparameter grid (lr ∈ {5e-5, 1e-4, 3e-4} × wd ∈ {0, 1e-4}),
the `wd=1e-4` runs at **lr=1e-4 and lr=5e-5** are bit-identical to their `wd=0` siblings —
per-fold validation loss matching to 15 decimal places across all 5 folds, confirmed from the
checkpoints themselves. Only the `lr=3e-4` pair shows a genuine wd=0 vs. wd=1e-4 difference.

The root cause was never diagnosed; it predates the current trainer, and the optimizer selection
in `tau-denseunet/scripts/train.py` as committed looks correct. The practical consequence:
**do not claim wd=1e-4 was tested at lr=1e-4 or lr=5e-5** — it was not, despite checkpoints
existing under those names. The grid winner (`lr1e-4_wd0`, used for all final DenseUNet models)
is the best of the genuinely-distinct configurations regardless, and all four differ by ~0.24%
relative, so no result changes. Closing the gap properly would need 10 fresh fold-trainings.

## What is, and isn't, committed

**Committed:** both models' `src/` (or `denseunet/`), `scripts/`, and `slurm/` trees; this
README and [RESULTS.md](RESULTS.md); and a **curated records subset** — for each of the 8
held-out-TEST variants, only the three CSVs behind the published tables
(`pooled_wholebrain_*`, `region_pooled_*`, `region_crosssubject_metrics_*`) plus that variant's
`metrics_*.md`. That is 32 files, not the full ~8,600-file records tree.

**Not committed, and why:**

| | Size | Reason |
|---|---|---|
| `data/` | 47 GB | ADNI patient data — redistribution is restricted by the ADNI Data Use Agreement |
| `results/checkpoints/` | 780 GB | Too large to distribute |
| `results/generated/` | 111 GB | Too large to distribute |
| `results/figures/` | 1.6 GB | Regenerable from the committed scripts |
| `per_subject_wholebrain.csv` | small | Joins ADNI RIDs to diagnostic group (CN/MCI/AD) — subject-level identifiers linked to clinical status, restricted under the ADNI DUA. The aggregate `pooled_*` / `region_*` files carry no RIDs and are unaffected. If per-subject data needs to be published, replace the RID with a sequential ID and keep the mapping private. |
| `*.bak*`, `*.pre*` | ~190 files | Backups left by in-place cluster patch scripts |

**Because checkpoints are not distributed, the committed records CSVs are the only verifiable
artifact in this repository.** Every number in [RESULTS.md](RESULTS.md) is recomputable from
them; nothing there is transcribed by hand.

**One script is permanently lost.** `make_per_subject_taugennet.py`, which generated the
TauGenNet per-subject metric CSVs, was destroyed by a mangled heredoc and is not recoverable.
Its *outputs* survive in the archive; the script does not. The DenseUNet-side equivalent
(`make_per_subject.py`) does survive and is committed under `analysis/`.

## Repository layout

```
taugennet/
  src/                    model, diffusion, dataset, conditioning, atlas
  scripts/                train / generate / evaluate_final / figure scripts
  slurm/                  cluster job scripts (train, eval, grids, CV, ablations)
  results/records/        curated held-out-TEST CSVs (see above)
tau-denseunet/
  denseunet/              model + config + mask utilities
  scripts/                train / generate / evaluate / aggregate
  slurm/                  cluster job scripts
  results/records/        curated held-out-TEST CSVs (see above)
analysis/                 figure and rescoring scripts used for the paper
RESULTS.md                held-out TEST tables, recomputed from the committed CSVs
```

## Environment

```bash
export TAUGENNET_ROOT=/path/to/this/repo/taugennet
export PYTHONPATH=$TAUGENNET_ROOT:${PYTHONPATH:-}
```

`TAUGENNET_ROOT` is read by `taugennet/src/config.py`. **If it is unset, paths silently fall
back to a stale hardcoded default and everything writes into the wrong place** — set it before
anything else. DenseUNet reads the same variable to locate the shared dataset layer.

Developed against Python 3.9. Dependencies (no pinned `requirements.txt` yet): `torch`,
`nibabel`, `numpy`, `scipy`, `scikit-image`, `nilearn`, `matplotlib`, `pandas`, `tqdm`, and
`transformers` (only for the `ptau217` / `combined` conditioning modes, which use a frozen text
encoder).

This repository contains no ADNI data. Both models expect a locally populated `data/raw/` from
your own ADNI pull; `taugennet/src/config.py` and `taugennet/src/dataset_final.py` document the
expected file layout.

## Quickstart — diffusion baseline (`taugennet/`)

```bash
cd taugennet

# Train (AE + diffusion) — --mode is the only required flag
python scripts/train.py --mode atrophy

# Evaluate: fresh inference from a checkpoint, cache the generations, score them
python scripts/evaluate_final.py --mode atrophy \
    --checkpoint-dir results/checkpoints/<your_run> \
    --save-generated --generated-dir results/generated/<variant>/atrophy

# Re-score cached generations later, no GPU needed
python scripts/evaluate_final.py --mode atrophy --use-cached \
    --generated-dir results/generated/<variant>/atrophy
```

`--mode` selects conditioning: `atrophy` (regional atrophy z-scores), `ptau217` (plasma
biomarker via text encoder), or `combined`. `evaluate_final.py` writes all three metric
families plus glass-brain figures by default (`--no-glass-brain` to skip).

Two conventions worth knowing:

- **Always give `--generated-dir` an isolated per-variant path.** Sharing one path across
  variants silently overwrites one variant's generations with another's.
- **Evaluate cached generations under the same split they were generated with.** Mixing splits
  produces numbers that look plausible and are wrong.

SLURM scripts for training, evaluation, grid search, CV folds, and ablations are under `slurm/`.

## Quickstart — DenseUNet baseline (`tau-denseunet/`)

```bash
cd tau-denseunet

python scripts/train.py    --mode atrophy
python scripts/generate.py --mode atrophy   # writes results/generated/atrophy/subject_*.npy

# Score with the shared evaluate_final.py — same pipeline, no duplicated metric code
cd $TAUGENNET_ROOT
python scripts/evaluate_final.py --mode atrophy --use-cached \
    --generated-dir <path-to>/tau-denseunet/results/generated/atrophy
```

See [`tau-denseunet/README.md`](tau-denseunet/README.md) for what this baseline ports verbatim
from upstream versus what was adapted.

## Status

This is not a frozen snapshot. Work still in flight at the time of this commit: 15 DenseUNet CV
training jobs with chained evaluation (which will add new records), a per-subject CV rescore,
and the RID zero-padding fix if it is ever applied. Expect follow-up commits.
