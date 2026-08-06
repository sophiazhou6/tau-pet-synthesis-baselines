#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""aggregate_recon.py — aggregate AE reconstruction across grid cells / folds.

Walks a checkpoint tree for `recon_metrics.json` files, groups them by the cell directory
(the parent of `fold_*`), and reports mean +/- std across folds.

    aggregate_recon.py results/checkpoints/grid_ae_struct
    aggregate_recon.py results/checkpoints/grid_search/step0

IMPORTANT — recon CANNOT select latent_ch / kl / depth.
`run_grid_search.py` states it directly: recon "always favours more channels + lower KL."
More AE capacity (more channels, more residual blocks, less downsampling) will essentially
always reconstruct better, while making the latent a HARDER target for the diffusion model.
Use this only as a DIAGNOSTIC (what is the AE's ceiling for this config?), never as a selector.
Select on generated output: scripts/localization_metrics.py --sort-by peak
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+", help="checkpoint dir(s) containing <cell>/fold_*/recon_metrics.json")
    p.add_argument("--metric", default="pearson", choices=["pearson", "ssim", "nrmse", "mse", "mae"])
    a = p.parse_args()

    cells = defaultdict(list)
    for root in a.roots:
        for f in glob.glob(os.path.join(root, "**", "recon_metrics.json"), recursive=True):
            # .../<cell>/fold_N/recon_metrics.json  -> cell
            fold_dir = os.path.dirname(f)
            cell = os.path.basename(os.path.dirname(fold_dir))
            try:
                cells[cell].append(json.load(open(f)))
            except Exception:
                pass

    if not cells:
        raise SystemExit("no recon_metrics.json found")

    higher_better = a.metric in ("pearson", "ssim")
    rows = []
    for cell, ms in cells.items():
        v = [m[a.metric] for m in ms if a.metric in m]
        if v:
            rows.append((cell, len(v), float(np.mean(v)), float(np.std(v))))
    rows.sort(key=lambda r: r[2], reverse=higher_better)

    w = max(len(r[0]) for r in rows) + 2
    print(f"{'cell':{w}s} {'folds':>5s}  recon {a.metric}")
    for cell, n, mu, sd in rows:
        print(f"{cell:{w}s} {n:5d}  {mu:.4f} +/- {sd:.4f}")

    print(f"\nranked by recon {a.metric} ({'higher' if higher_better else 'lower'} = better).")
    print("WARNING: recon rewards AE capacity and cannot choose latent_ch/kl/depth.")
    print("         Treat these as CEILINGS, not a ranking. Select on PEAK-region R of")
    print("         generated output (scripts/localization_metrics.py --sort-by peak).")


if __name__ == "__main__":
    main()
