# TauGenNet

## Project Context

This is a medical imaging research project implementing TauGenNet — a 3D diffusion model for generating tau PET brain scans conditioned on MRI.
- Input resolution: 96×112×96
- Sampler: DDPM (500 steps), not DDIM
- Conditioning: MRI (structural) + optional demographic/biomarker variables

## Environment

- Claude runs on Sophia's Mac; all commands run on **UCSF CoreHPC over SSH** (cluster user `adebnath1`).
  There is no persistent cwd between pasted SSH blocks — always prefix commands with `cd ~/taugennet`.
- Repo: `~/taugennet` → symlink to `/mnt/scratch/user/adebnath1/taugennet` (**scratch purges ~30 days** —
  pull figures/records off promptly).
- **Interactive preamble — run before anything else**, or `src/config.py:56` silently falls back to a
  stale Princeton path and figures/records write into the void:
  ```bash
  cd ~/taugennet
  export TAUGENNET_ROOT=$HOME/taugennet
  export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages:$HOME/taugennet:${PYTHONPATH:-}
  ```
- Do not assume Colab or a cloud notebook environment
- Resolve imports and kernels against the project venv

## Data & Reproducibility

Before stating any data statistics (split sizes, subject counts, ratios), read the config and dataset files directly — do not infer or estimate. Cite the file and line when reporting these numbers. Key facts to verify from source:
- Dataset: ADNI tau PET + MRI, 96×112×96 voxels; subject counts vary by mode (read from logs or dataset at runtime)
- Always distinguish per-subject vs. global statistics explicitly

## Evaluation & Figures

When generating evaluation figures or tables:
- Include all standard metrics: NRMSE, MAE, MSE, SSIM, Pearson
- Clearly label whether statistics are global or per-subject
- Prioritize per-subject calculations and voxel-wise metrics

When reporting metrics or figures, clearly distinguish global vs per-subject statistics and include all requested metrics (e.g., MSE, SSIM, Pearson r). Do not omit a requested metric.

### Canonical eval — ALWAYS produce these (default, do not re-derive by hand)
The three SUVR Pearson sets + glass brain are the standard deliverable. `evaluate_final.py`
computes all three in **SUVR (unnormalized)** by default; the reusable figure script is
`scripts/suvr_eval_figures.py` (SET1 hexbin, SET2 bar, SET3 bar, SET2-vs-SET3 scatter + glass brain).
- **SUVR reconstruction:** unnormalize with the dataset's `get_pet_norms` (masked min/max).
  `dataset_final/combined/spatial` all expose it. NEVER fall back to raw-NIfTI min/max — the raw max
  includes out-of-brain hot voxels (~20 vs true in-brain ~5-7) and inflates SET1/SET3 ~1.5-2.4x
  (e.g. silu combined SET3 0.71 buggy → 0.29 correct). True regional cross-subject R here is ~0.3-0.4.
- **Split:** evaluate cached gens with the SAME split they were generated under (all existing gens =
  legacy → pass `--no-use-mentor-split`). Mentor split is for NEW training only.
- **Always cache generations:** every run must `--save-generated` to an explicit, isolated
  `--generated-dir results/generated/<variant>/<mode>[/fold_N]` (never share a path across variants —
  the old collision bug saved cond_a into silu/atrophy).

### Canonical metric sets (`evaluate_final.py`)
Three distinct families, all in **SUVR (unnormalized)** space, DK86-masked:
1. **Pooled voxel-level, whole brain** — all in-brain voxels of all subjects pooled into one
   real/gen vector → single R/NRMSE/MSE/MAE + per-subject-averaged SSIM. CSV: `pooled_wholebrain_{mode}.csv`.
2. **Pooled voxel-level, per region (86 DK regions)** — same pooling, restricted to each atlas
   region. CSV: `region_pooled_{mode}.csv` (86 rows).
3. **Regional cross-subject (86 DK regions)** — average each region's voxels per subject (N×86),
   then correlate real-vs-gen regional means across subjects. CSVs: `regional_means_real_{mode}.csv`
   and `regional_means_gen_{mode}.csv` (rows = subjects, cols = 86 regions), plus
   `region_crosssubject_metrics_{mode}.csv`.

Region labels come from the DK86 atlas (`atlas_labels.load_atlas`, labels 1-86), nearest-neighbor
interpolated to `VOL_SHAPE`. **SSIM is reported at whole-brain level only** — it is windowed/spatial,
so a single region's bounding box leaks neighboring tissue; the 86-region tables omit SSIM
(`REGION_METRIC_KEYS`).

## Code Reuse

Before building new utilities (e.g., decomposition, evaluation scripts), check for existing project libraries/scripts (`decompose_lib`, `evaluate_final.py`) and reuse them instead of reimplementing.

## Canonical Pipeline

### Environment Setup
- Python interpreter: `/usr/bin/python3.9` — **never** `python` or the old Princeton conda path.



### Training Launch
```bash
# Full paper pipeline (AE + 3 diffusion modes, ~2 days on A100)
sbatch train_paper.slurm

# Custom split
python scripts/train.py --mode atrophy --dataset v2 --split 80/10/10 \
    --ae-checkpoint results/checkpoints/taugennet_checkpoint_paper.pt --skip-ae \
    --diff-epochs 2000 --patience 500

# End-to-end orchestrator (submits, polls, evaluates, updates README, stages commit)
python scripts/run_experiment.py
python scripts/run_experiment.py --split 80/10/10          # custom split
python scripts/run_experiment.py --from-stage generate_tables  # resume/rerun from a stage
```

### Evaluation Launch
```bash
# Canonical eval (fresh inference): batch SLURM scripts pass --checkpoint-dir
sbatch slurm/eval/eval_final_silu.slurm     # silu, all 3 modes; sbatch ...relu.slurm for relu

# Interactive (reuse cached generations, no GPU):
python scripts/evaluate_final.py --mode atrophy --generated-dir results/generated/silu/atrophy
```
- `--use-cached` default is **context-aware**: cached unless `--checkpoint-dir` is given
  (batch scripts → fresh inference; interactive runs → cache). Force with `--use-cached`/`--no-use-cached`.
- `--glass-brain` defaults **on** (renders glass/slice/scatter via `glass_brain_final.run_mode`
  on in-memory volumes — no cache reload; needs nilearn). Skip with `--no-glass-brain`.

### Job Monitoring
```bash
squeue -u adebnath1
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed
tail -f results/logs/taugennet_<JOBID>.out
```
### Always check
Never overwrite any existing files.
Always use the data mask that blocks out 0.

### CLI scripts
Always use `/usr/bin/python3.9` not `python` when running a script.
### Git Commit Conventions
- Prefix: `feat:` / `fix:` / `eval:` / `chore:` / `exp:`
- Include mode and split when relevant: `eval: paper metrics atrophy/ptau217/combined (70/10/20)`
- **Never commit:** `results/checkpoints/`, `data/`, `venv/`, `results/figures/`
- **Always commit after eval:** `results/records/metrics.md`, `README.md`
