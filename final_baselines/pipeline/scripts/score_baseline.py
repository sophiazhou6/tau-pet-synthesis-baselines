#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Score cached predictions of one run/fold with the lab's evaluation pipeline from
github.com/sophiazhou6/tau-pet-synthesis-baselines (--pipeline github, the default):

  1. denseunet/gh_align.py (under the GitHub taugennet root): rebuilds the held-out test set
     with the GitHub loader, verifies it equals the manifest's heldout_test rows in the same
     order, and writes per_subject_wholebrain.csv (analysis/make_per_subject.py convention).
  2. taugennet/scripts/evaluate_final.py --use-cached, invoked exactly as
     tau-denseunet/scripts/evaluate.py does (--mode atrophy, --skip-ablation, --fold/--n-folds,
     --metrics-out, absolute --figures-dir/--records-dir, glass brains on): writes
     pooled_wholebrain_atrophy.csv (SET1), region_pooled_atrophy.csv (SET2),
     region_crosssubject_metrics_atrophy.csv (SET3), regional_means_{real,gen}_atrophy.csv,
     metrics.json (per-subject whole-brain, for aggregate_cv_eval.py) and all figures.
  3. taugennet/scripts/localization_metrics.py functions on the regional-mean CSVs →
     localization.json (SET3 / WITHIN / PEAK@K), and a summary block appended to metrics.json.

--pipeline local is the in-process fallback (same metric functions imported from the local
taugennet copy; used by the smoke test's --limit-subjects runs, which the GitHub loader
cannot subset).

    python scripts/score_baseline.py --model swinunetr --manifest mentor235 --fold 0
    python scripts/score_baseline.py --model swinunetr --manifest cohort_v2 --fold 0 --use-controls-322 \\
        --controls-dir $TAUGENNET_ROOT/data/raw_v2/controls_322      # cohort materialised by prepare_cohort.py
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import (FIGURES_ROOT, GEN_ROOT, GH_TAUGENNET, N_FOLDS, RECORDS_ROOT,  # noqa: E402
                              REPO_ROOT, TAUGENNET_ROOT)
from denseunet import predictions as preds  # noqa: E402
from denseunet.cohort import read_manifest  # noqa: E402
from train_baseline import cohort_name, resolve_manifest, run_dir  # noqa: E402

LOCAL_FILES = ["pooled_wholebrain_SUVR.csv", "region_pooled_SUVR.csv", "region_crosssubject_metrics_SUVR.csv",
               "regional_means_real_SUVR.csv", "regional_means_gen_SUVR.csv", "per_subject_metrics.csv"]


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ───────────────────────────── GitHub pipeline ─────────────────────────────

def score_github(args, gen_dir, out_dir, fig_dir):
    if not os.path.exists(os.path.join(GH_TAUGENNET, "scripts", "evaluate_final.py")):
        sys.exit(f"GitHub taugennet not found at {GH_TAUGENNET} (set TAU_GH_ROOT or clone "
                 f"github.com/sophiazhou6/tau-pet-synthesis-baselines and symlink taugennet/data/raw).")
    env = dict(os.environ, TAUGENNET_ROOT=GH_TAUGENNET, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg",
               PYTHONPATH=f"{GH_TAUGENNET}:{os.path.join(GH_TAUGENNET, 'scripts')}:{os.environ.get('PYTHONPATH', '')}")
    fold_args = [] if args.fold is None else ["--fold", str(args.fold), "--n-folds", str(args.n_folds)]
    cohort_args = (["--use-controls-322"] + (["--controls-dir", args.controls_dir] if args.controls_dir else [])
                   if args.use_controls_322 else [])

    print(f"[score/github] 1/3 alignment check + per-subject table (GitHub loader @ {GH_TAUGENNET})", flush=True)
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "denseunet", "gh_align.py"), "--manifest", args.manifest,
                    "--gen-dir", gen_dir, "--out-dir", out_dir, "--split", args.split] + fold_args + cohort_args,
                   env=env, check=True)

    print("[score/github] 2/3 evaluate_final.py --use-cached (the canonical evaluator)", flush=True)
    cmd = [sys.executable, os.path.join(GH_TAUGENNET, "scripts", "evaluate_final.py"), "--mode", "atrophy",
           "--use-cached", "--generated-dir", gen_dir, "--figures-dir", fig_dir, "--records-dir", out_dir,
           "--skip-ablation", "--metrics-out", os.path.join(out_dir, "metrics.json"),
           "--glass-brain" if args.glass_brain else "--no-glass-brain", "--eval-on", args.split] \
        + fold_args + cohort_args + args.eval_extra
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True, cwd=GH_TAUGENNET)

    print("[score/github] 3/3 localization_metrics.py (SET3 / WITHIN / PEAK)", flush=True)
    loc = _load_module(os.path.join(GH_TAUGENNET, "scripts", "localization_metrics.py"), "gh_localization")
    Rv, Gv = loc._load_pair(out_dir)
    if Rv is None:
        sys.exit(f"no regional_means_*.csv written by evaluate_final in {out_dir}")
    s3, within, peak, n_subj, n_reg = loc.metrics(Rv, Gv, args.top_k)
    localization = {"set3": float(s3), "within": float(within), f"peak{args.top_k}": float(peak),
                    "n_subj": int(n_subj), "n_regions": int(n_reg), "top_k": args.top_k}
    with open(os.path.join(out_dir, "localization.json"), "w") as f:
        json.dump(localization, f, indent=2)

    # summary block: SET1 from the pooled CSV row, SET2/3 = mean pearson over the 86 regions (RESULTS.md recipe)
    set1 = pd.read_csv(os.path.join(out_dir, "pooled_wholebrain_atrophy.csv"), index_col=0).iloc[0]
    set2 = pd.read_csv(os.path.join(out_dir, "region_pooled_atrophy.csv"), index_col=0)
    set3 = pd.read_csv(os.path.join(out_dir, "region_crosssubject_metrics_atrophy.csv"), index_col=0)
    per_subj = pd.read_csv(os.path.join(out_dir, "per_subject_wholebrain.csv"))
    mpath = os.path.join(out_dir, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
    metrics.update({
        "tag": args.tag, "cohort": cohort_name(args.manifest), "fold": args.fold, "pipeline": "github",
        "gh_root": GH_TAUGENNET, "n": int(len(per_subj)),
        "set1_pooled_suvr": {k: float(set1[k]) for k in set1.index if k in ("pearson", "nrmse", "ssim", "mse", "mae")},
        "set2_region_pooled_pearson_mean": float(np.nanmean(set2["pearson"])),
        "set3_region_crosssubject_pearson_mean": float(np.nanmean(set3["pearson"])),
        "within_r_mean": float(within), f"peak{args.top_k}_r_mean": float(peak),
        "per_subject_suvr": {k: float(per_subj[k].mean()) for k in ("pearson_wholebrain", "mse_wholebrain", "rmse_wholebrain", "mae_wholebrain")},
        "per_group_pearson_suvr": {g: float(v) for g, v in per_subj.groupby("group")["pearson_wholebrain"].mean().items()},
    })
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ───────────────────────────── local fallback ─────────────────────────────

def _safe_r(a, b):
    from scipy.stats import pearsonr
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(pearsonr(a, b)[0])


def score_local(args, gen_dir, out_dir, fig_dir):
    from denseunet.cohort import build_cohort
    ef = _load_module(os.path.join(TAUGENNET_ROOT, "scripts", "evaluate_final.py"), "evaluate_final_mod")
    from src.dataset_final import unnormalize  # noqa: E402
    from denseunet import figs  # noqa: E402

    data = build_cohort(args.manifest, fold_idx=args.fold, n_folds=args.n_folds, batch_size=1, num_workers=0,
                        limit_subjects=args.limit_subjects)
    test_ds = data["test_ds"]
    mask, bbox, label_vol = ef._dk86_mask(test_ds), None, ef._dk86_label_volume(test_ds)
    bbox = ef._mask_bbox(mask)
    real_norm, gen_masked, real_suvr, gen_suvr = [], [], [], []
    for i in range(len(test_ds)):
        fp = os.path.join(gen_dir, f"subject_{i:03d}.npy")
        if not os.path.exists(fp):
            sys.exit(f"missing {fp} (test set has {len(test_ds)} subjects)")
        real = test_ds[i][0].squeeze().numpy()
        gen = np.load(fp).astype(np.float32)
        pmin, pmax = test_ds.get_pet_norms(i)
        real_norm.append(real); gen_masked.append(gen * mask)
        real_suvr.append(unnormalize(real, pmin, pmax)); gen_suvr.append(unnormalize(gen, pmin, pmax) * mask)
    keys = ["pearson", "nrmse", "ssim", "mse", "mae"]
    pn = ef.compute_wholebrain_metrics(real_norm, gen_masked, brain_mask=mask, bbox=bbox)
    ps = ef.compute_wholebrain_metrics(real_suvr, gen_suvr, brain_mask=mask, bbox=bbox)
    per_subject = test_ds.index_table()[["index", "RID", "group"]].copy()
    for k, vn, vs in zip(keys, pn, ps):
        per_subject[f"{k}_norm"] = vn; per_subject[f"{k}_suvr"] = vs
    set1 = ef.compute_pooled_wholebrain_metrics(real_suvr, gen_suvr, mask)
    set2 = ef.compute_region_pooled_metrics(real_suvr, gen_suvr, label_vol)
    set3, Rreal, Rgen = ef.compute_region_crosssubject_metrics(real_suvr, gen_suvr, label_vol)
    ef.save_canonical_metric_sets(set1, set2, set3, Rreal, Rgen, "SUVR", out_dir)
    within, peak = [], []
    for i in range(Rreal.shape[0]):
        r, g = Rreal[i], Rgen[i]
        within.append(_safe_r(r, g))
        idx = np.argsort(np.where(np.isfinite(r), r, -np.inf))[-args.top_k:]
        peak.append(_safe_r(r[idx], g[idx]))
    per_subject["within_r"] = within; per_subject[f"peak{args.top_k}_r"] = peak
    per_subject.to_csv(os.path.join(out_dir, "per_subject_metrics.csv"), index=False)
    metrics = {"tag": args.tag, "cohort": cohort_name(args.manifest), "fold": args.fold, "n": len(test_ds),
               "pipeline": "local", **{k: float(np.nanmean(v)) for k, v in zip(keys, pn)},
               "suvr_persubject": {k: float(np.nanmean(v)) for k, v in zip(keys, ps)},
               "set1_pooled_suvr": {k: float(set1[k].iloc[0]) for k in set1.columns},
               "set2_region_pooled_pearson_mean": float(np.nanmean(set2["pearson"])),
               "set3_region_crosssubject_pearson_mean": float(np.nanmean(set3["pearson"])),
               "within_r_mean": float(np.nanmean(within)), f"peak{args.top_k}_r_mean": float(np.nanmean(peak))}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    r_all = np.concatenate([r[mask].ravel() for r in real_suvr]); g_all = np.concatenate([g[mask].ravel() for g in gen_suvr])
    figs.plot_suvr_figures(r_all, g_all, float(set1["pearson"].iloc[0]), set2, set3, os.path.join(fig_dir, "suvr"),
                           tag=f"({args.tag}, fold {args.fold})")
    if args.glass_brain:
        try:
            sys.path.insert(0, os.path.join(TAUGENNET_ROOT, "scripts"))
            from glass_brain_final import run_mode  # noqa: E402
            run_mode("atrophy", args.tag, os.path.join(fig_dir, "glass_brain"),
                     data=(real_suvr, gen_suvr, real_norm, gen_masked, test_ds), rank_by="ssim")
        except Exception as e:
            print(f"[score] glass-brain rendering failed: {type(e).__name__}: {e}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="only used to derive default dirs/tag")
    p.add_argument("--manifest", required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--tag", default=None)
    p.add_argument("--split", choices=["test", "val"], default="test",
                   help="val: score this fold's validation predictions (<run>_val) — the CV column")
    p.add_argument("--pipeline", choices=["github", "local"], default="github")
    p.add_argument("--use-controls-322", action="store_true",
                   help="github pipeline: the cohort is a controls_322-style layout (CN+MCI+AD), see prepare_cohort --materialize")
    p.add_argument("--controls-dir", default=None, help="github pipeline: dir with the numeric split CSVs (default data/raw/controls_322)")
    p.add_argument("--eval-extra", nargs="*", default=[], help="extra flags forwarded verbatim to evaluate_final.py")
    p.add_argument("--generated-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--fig-dir", default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--glass-brain", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit-subjects", type=int, default=None, help="local pipeline only (debug)")
    p.add_argument("--pack", action=argparse.BooleanOptionalAction, default=True,
                   help="after a successful score, replace subject_*.npy by one lossless predictions.npz (~11x smaller)")
    args = p.parse_args()
    args.tag = args.tag or args.model
    if not args.tag:
        sys.exit("give --model or --tag")
    args.manifest = resolve_manifest(args.manifest)
    if args.pipeline == "github" and not args.use_controls_322 \
            and "CN" in set(read_manifest(args.manifest, check_files=False)["group"]):
        # the GitHub loader only sees CN subjects on its controls_322 path, so without this flag a
        # CN-containing manifest can never pass the alignment check
        print("[score] manifest contains CN subjects → enabling --use-controls-322", flush=True)
        args.use_controls_322 = True
    sfx = "_val" if args.split == "val" else ""
    if args.split == "val" and (args.fold is None or args.pipeline != "github"):
        sys.exit("--split val needs --fold and the github pipeline")
    gen_dir = args.generated_dir or run_dir(GEN_ROOT, args.manifest, args.tag, args.fold) + sfx
    out_dir = args.out_dir or run_dir(RECORDS_ROOT, args.manifest, args.tag, args.fold) + sfx
    fig_dir = args.fig_dir or run_dir(FIGURES_ROOT, args.manifest, args.tag, args.fold) + sfx
    os.makedirs(out_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)
    if not preds.has_predictions(gen_dir):
        sys.exit(f"No cached predictions in {gen_dir}")
    if args.limit_subjects and args.pipeline == "github":
        print("[score] --limit-subjects only works with --pipeline local; switching."); args.pipeline = "local"

    # packed run (predictions.npz only) → unpack to node-local temp space for the evaluator
    work_dir, tmp_dir = gen_dir, None
    if not preds.npy_files(gen_dir):
        tmp_dir = work_dir = preds.unpack(gen_dir)
        print(f"[score] unpacked {preds.PACK_NAME} → {tmp_dir}")
    try:
        metrics = score_github(args, work_dir, out_dir, fig_dir) if args.pipeline == "github" \
            else score_local(args, work_dir, out_dir, fig_dir)
    finally:
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
    if args.pack and tmp_dir is None:
        preds.pack(gen_dir)

    s1 = metrics["set1_pooled_suvr"]
    print(f"\n[score/{metrics['pipeline']}] {args.tag} fold {args.fold}  n={metrics['n']}")
    print(f"  per-subject whole-brain (metrics.json): Pearson {metrics.get('pearson', float('nan')):.4f}  "
          f"NRMSE {metrics.get('nrmse', float('nan')):.4f}  SSIM {metrics.get('ssim', float('nan')):.4f}  "
          f"MSE {metrics.get('mse', float('nan')):.5f}  MAE {metrics.get('mae', float('nan')):.5f}")
    print(f"  SET1 pooled SUVR    R {s1['pearson']:.4f}  NRMSE {s1['nrmse']:.4f}  MSE {s1['mse']:.4f}  "
          f"MAE {s1['mae']:.4f}  SSIM {s1.get('ssim', float('nan')):.4f}")
    print(f"  SET2 R {metrics['set2_region_pooled_pearson_mean']:.4f}   SET3 R "
          f"{metrics['set3_region_crosssubject_pearson_mean']:.4f}   WITHIN {metrics['within_r_mean']:.4f}   "
          f"PEAK@{args.top_k} {metrics[f'peak{args.top_k}_r_mean']:.4f}")
    print(f"  → {out_dir}\n  → {fig_dir}")


if __name__ == "__main__":
    main()
