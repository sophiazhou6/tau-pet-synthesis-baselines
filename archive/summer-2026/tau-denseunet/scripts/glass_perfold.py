#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Per-fold glass brains (NO ensemble averaging).

Each of the 5 fold-models predicts the same fixed 47 mentor-test subjects, so there
are 47 x 5 = 235 individual predictions. This pools all 235 as separate items and
lets glass_brain_final.run_mode pick Best / Median / Worst among them — once ranked
by MSE, once by SSIM. Every prediction is displayed over the DK86 region.

  --mode atrophy   → both variants:
     DK86 model         → results/figures/atrophy/glass_brain_perfold/{mse,ssim}/
     whole-brain model  → results/figures/atrophy_wb_on_dk86/glass_brain_perfold/{mse,ssim}/

The full 235-item ranking (which subject+fold is best/median/worst) is written to
results/records/perfold_rank_<mode>.json for reference.
"""

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import TAUGENNET_ROOT, REPO_ROOT, GENERATED_DIR, N_FOLDS, build_dataloaders  # noqa: E402
from denseunet.masks import load_or_build_wholebrain_mask, inject_mask  # noqa: E402

FIGURES_DIR = os.path.join(REPO_ROOT, "results", "figures")
RECORDS_DIR = os.path.join(REPO_ROOT, "results", "records")


def _load(name):
    for pth in (TAUGENNET_ROOT, os.path.join(TAUGENNET_ROOT, "scripts")):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    path = os.path.join(TAUGENNET_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"{name}_mod", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def build_pool(mode, mask_mode, ef, unnormalize, n_folds):
    """Return the 235-item pool ordered subject-major: index i*n_folds + f."""
    _, _, ds, *_ = build_dataloaders(mode=mode, use_dk_mask=True)
    if mask_mode == "wholebrain":
        inject_mask(load_or_build_wholebrain_mask(mode), ds)
    dk = ef._atlas_dk86_binary()                       # DK86 display region (both variants)
    run_tag = mode if mask_mode == "dk86" else f"{mode}_wholebrain"

    # per-subject real (SUVR + normalized) and norms — fold-independent
    real_suvr, real_norm, norms = [], [], []
    for i in range(len(ds)):
        r = ds[i][0].squeeze().numpy()
        pmin, pmax = ds.get_pet_norms(i)
        real_suvr.append(unnormalize(r, pmin, pmax) * dk)
        real_norm.append(r * dk); norms.append((pmin, pmax))

    real_pool, gb_gen_pool, real_n_pool, gen_n_pool, tags = [], [], [], [], []
    for i in range(len(ds)):
        for f in range(n_folds):
            p = os.path.join(GENERATED_DIR, run_tag, f"fold_{f}", f"subject_{i:03d}.npy")
            gen = np.load(p).astype(np.float32)
            gs = unnormalize(gen, *norms[i]) * dk
            real_pool.append(real_suvr[i])
            gb_gen_pool.append(gs * (real_suvr[i] > 0))
            real_n_pool.append(real_norm[i]); gen_n_pool.append(gen * dk)
            tags.append((i, f))
    return ds, (real_pool, gb_gen_pool, real_n_pool, gen_n_pool), tags


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["atrophy", "ptau217"], default="atrophy")
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = p.parse_args()

    ef = _load("evaluate_final")
    run_mode = _load("glass_brain_final").run_mode
    from src.dataset_final import unnormalize  # noqa: E402

    VARIANTS = [("dk86", "atrophy"), ("wholebrain", "atrophy_wb_on_dk86")]
    summary = {}
    for mask_mode, display_dir in VARIANTS:
        ds, data, tags = build_pool(args.mode, mask_mode, ef, unnormalize, args.n_folds)
        real_n, gen_n = data[2], data[3]
        # per-item metrics (normalized) for the reference ranking
        gb = _load("glass_brain_final")
        m = [gb._compute_metrics(r, g) for r, g in zip(real_n, gen_n)]

        gb_dir = os.path.join(FIGURES_DIR, display_dir, "glass_brain_perfold")
        for rank in ("mse", "ssim"):
            run_mode(args.mode, "silu", gb_dir, use_dk_mask=True, data=(*data, ds), rank_by=rank)

        # reference ranking table (subject, fold) → sorted
        def rep(idx):
            i, f = tags[idx]; return {"subject": i, "fold": f, **{k: float(m[idx][k]) for k in ("mse", "ssim")}}
        order_mse = np.argsort([mm["mse"] for mm in m])       # ascending (best first)
        order_ssim = np.argsort([-mm["ssim"] for mm in m])    # descending (best first)
        n = len(m)
        summary[display_dir] = {
            "n_items": n, "note": "235 per-fold predictions ranked as individual items",
            "by_mse":  {"best": rep(int(order_mse[0])),  "median": rep(int(order_mse[n // 2])),  "worst": rep(int(order_mse[-1]))},
            "by_ssim": {"best": rep(int(order_ssim[0])), "median": rep(int(order_ssim[n // 2])), "worst": rep(int(order_ssim[-1]))},
        }
        b = summary[display_dir]
        print(f"\n[{display_dir}] {n} per-fold predictions:")
        print(f"  by MSE  — best subj{b['by_mse']['best']['subject']}/fold{b['by_mse']['best']['fold']} "
              f"(MSE={b['by_mse']['best']['mse']:.5f}); worst subj{b['by_mse']['worst']['subject']}/fold{b['by_mse']['worst']['fold']} "
              f"(MSE={b['by_mse']['worst']['mse']:.5f})")
        print(f"  by SSIM — best subj{b['by_ssim']['best']['subject']}/fold{b['by_ssim']['best']['fold']} "
              f"(SSIM={b['by_ssim']['best']['ssim']:.4f}); worst subj{b['by_ssim']['worst']['subject']}/fold{b['by_ssim']['worst']['fold']} "
              f"(SSIM={b['by_ssim']['worst']['ssim']:.4f})")
        print(f"  → {gb_dir}/{{mse,ssim}}/")

    json.dump(summary, open(os.path.join(RECORDS_DIR, f"perfold_rank_{args.mode}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
