# Dense-U-Net grid search — runbook (new server)

Order of operations to reproduce the DK86 hyperparameter grid search from the
transfer bundle. Grid = lr {5e-5, 1e-4, 3e-4} x weight_decay {0, 1e-4} = 6 configs
x 5 folds = 30 runs, 126 fixed epochs each (early stopping OFF). Selection is by
mean validation loss; the 47-subject test set is never used to pick the winner.

Everything runs from the baseline repo root unless noted. `PY` = the taugennet
conda python on the new server.

---

## Part 0 — one-time setup (do once, in this order)

1. **taugennet dependency** — extract the v2 bundle as `<TAUGENNET>` (gives
   `src/`, `scripts/evaluate_final.py`, `glass_brain_final.py`,
   `aggregate_cv_eval.py`, `data/raw/{heldout_test_split,train_val_split}.csv`,
   atlas, ADNI CSVs). Extract the **v3 bundle** too for
   `scripts/localization_metrics.py` (PEAK@10).

2. **Data** — make sure the ADNI volumes are under `<TAUGENNET>/data/raw/`
   (`cerebellumNormalized_AD_MCI/…`, `1mm_parcellated_*_subj/…`). Segmentation
   files are NOT needed for the DK86 grid.

3. **Baseline code** — extract `baseline_transfer_code.tar.gz` as `<REPO>`.

4. **Conda env** — recreate the env (torch, nibabel, nilearn, scikit-image,
   scipy, scikit-learn, pandas, matplotlib). Note its python path as `PY`.

5. **Make `src` importable** — `pip install -e <TAUGENNET>` (or ensure
   `TAUGENNET_ROOT` is on `sys.path`; `denseunet/config.py` adds it).

6. **Fix the Adroit-hardcoded paths:**
   - `export TAUGENNET_ROOT=<TAUGENNET>` (config.py reads this env var).
   - In `slurm/*.slurm`: `PYTHON=` → `PY`; `REPO=` → `<REPO>`; and the
     `#SBATCH --output/--error` absolute log paths.
   - In `slurm/grid_search.slurm`: `#SBATCH --partition`, `--gres`, `--mail-user`
     for the new cluster (+ any `module load` / `conda activate`).
   - In `scripts/aggregate_cv.py`: `PY = "…python3"` (only needed for scoring, not
     the grid itself).

---

## Part 1 — verify BEFORE submitting (cheap, CPU-only)

7. **Split sanity** — confirm the mentor split loads (47 test / 188 dev, folds
   partition the dev pool):
   ```bash
   CUDA_VISIBLE_DEVICES="" $PY - <<'EOF'
   import csv, os
   from src.dataset_final import build_dataloaders
   RAW=os.path.join(os.environ["TAUGENNET_ROOT"],"data/raw")
   L=lambda p:{str(int(float(r["RID"]))) for r in csv.DictReader(open(os.path.join(RAW,p)))}
   TEST,DEV=L("heldout_test_split.csv"),L("train_val_split.csv")
   rid=lambda p:os.path.basename(os.path.dirname(p)).replace("RID_","")
   for i in range(5):
       _,_,te,*_=build_dataloaders(mode="atrophy",fold_idx=i,n_folds=5,use_dk_mask=False)
       assert {rid(p) for p in te.pet_paths}==TEST, f"fold {i} test != 47 heldout"
   print("OK: split verified, test fixed at 47 heldout")
   EOF
   ```

8. **Confirm fixed epochs** — `grep EPOCHS= slurm/grid_search.slurm` → should be
   `EPOCHS=126` (median best-epoch of the 5 DK86 folds). Change deliberately if
   desired; early stopping stays OFF so all configs share the budget.

---

## Part 2 — run the grid (GPU)

9. Submit the 30-task array (6 configs x 5 folds):
   ```bash
   sbatch slurm/grid_search.slurm atrophy dk86
   ```
   Each task trains one (config, fold) for 126 epochs and saves the best-val
   checkpoint to `results/checkpoints/atrophy/grid/lr<LR>_wd<WD>/fold_<f>/denseunet_best.pt`.
   ~1h/task. Wait for all 30 to finish (`squeue -u <user>`).

---

## Part 3 — pick the winner (by VALIDATION, unbiased)

10. Rank configs by mean validation loss across folds:
    ```bash
    $PY scripts/grid_summary.py --mode atrophy
    ```
    Prints each config's mean±std val loss and the WINNER. Do **not** pick by
    peeking at test.

---

## Part 4 — report the winner on the TEST set (the final number)

11. For the winning config `lr<LR>_wd<WD>`, cache its 5 fold-models' predictions
    on the fixed 47 test, then score + aggregate (reuses the standard pipeline):
    ```bash
    WIN=lr<LR>_wd<WD>                      # from step 10
    for f in 0 1 2 3 4; do
      $PY scripts/generate.py --mode atrophy --fold $f \
          --checkpoint results/checkpoints/atrophy/grid/$WIN/fold_$f/denseunet_best.pt \
          --out-dir results/generated/atrophy_tuned/fold_$f --overwrite
      $PY scripts/suvr_sets.py --mode atrophy --fold $f \
          --generated-dir results/generated/atrophy_tuned/fold_$f
    done
    # CV mean ± std across the 5 folds:
    $PY $TAUGENNET_ROOT/scripts/aggregate_cv_eval.py results/records/cv_atrophy
    $PY scripts/aggregate_suvr_sets.py --mode atrophy
    ```
    (Note: step 11 writes into the `cv_atrophy` records; move/rename the earlier
    baseline `cv_atrophy` first if you want to keep both.)

Report the winner's step-11 numbers (5-fold mean ± std on the 47 test) as the
tuned baseline result.
