# Hyperparameter sweep — CFG-enabled, ranked on SET1/2/3 + WITHIN + PEAK

**Run this AFTER the CFG conditioning matrix finishes.** Conditioning is the bigger lever
(it's the bottleneck), so pick the winning conditioning first, then sweep AE/arch under it.
Set `COND_MODE=` at the top of the diffusion scripts to that winner.

Every cell now: trains with **CFG** (`--cfg-prob 0.15`), evaluates with **guidance w=10 + K=16
posterior mean** (the best recipe found), and ranks on **all five metrics**
(SET1 / SET2 / SET3 / WITHIN / PEAK) via `peak_from_gens.py`. This matches how you'll deploy.

## Stage 1 — the interaction grid (5 AE + 10 diffusion)
`latent_ch {2,3,4,5,6}` x `UNet {small, large}`. latent_ch and capacity interact, so full grid.
```bash
bash slurm/sweep/submit_stage1.sh      # AEs, each chaining its 2 CFG diffusion arms
bash slurm/sweep/rank_sweep.sh <COND_MODE>   # SET1/2/3 + WITHIN + PEAK across all cells
```
- AE cells (`s1_ae_*`) are unconditional -> unchanged, no CFG.
- Diffusion cells (`s1_diff_lch*_{small,large}`) are FINAL-family (`train_cfg`). If the matrix
  winner is a spatial-MAP conditioning, use `s1_diff_SPATIAL_template.slurm` (spatial-family CFG)
  instead — copy per latent_ch/arm and set LCH/CH/NT/COND_MODE.

## Stages 2 & 3 — coordinate descent on the winner
```bash
bash slurm/sweep/gen_stage23.sh <winning_latent_ch> <small|large>
```
- Stage 2 `kl {1e-3,1e-4,1e-6}`: AE term -> fresh AE each (1e-5 already done in stage 1).
- Stage 3 `noise_schedule cosine`: diffusion-only, reuses the winning AE.
Both now train with CFG and eval with guidance, same as stage 1.

## Selection rule
Rank on **PEAK** for localization, but report **SET1/2/3** (your mandated Pearsons) too — they
can disagree (CFG's PEAK optimum is higher-w than its SET3 optimum). Watch NRMSE/contrast next to
PEAK: PEAK is per-subject scale-invariant and can't see over-smoothing on its own.

## Non-negotiables baked in
- **`--batch-size 4` on every AE job** (rb=2 OOMs an 80 GB A100 at batch 8).
- Never rank AEs by reconstruction (recon favours more channels + lower KL).
- CFG needs `--cfg-prob>0` at train time; guidance on a non-CFG checkpoint = garbage (the scripts warn).
