#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Redraw panel B (SET1 hexbin) with the run's actual inputs in its title. score_baseline.py always calls
evaluate_final.py with --mode atrophy, so the scorer's own plot says "atrophy" whatever the model used.
Same plotting function and same voxels; the R and voxel count are checked against the scored values.

    python scripts/replot_pooled_scatter.py --run denseunet_paper_refit --inputs "MRI only"
    python scripts/replot_pooled_scatter.py --run taugennet_paper --inputs "MRI + plasma p-tau217"

Writes <figures>/<cohort>/<tag>/<fold>/pooled_wholebrain_scatter_labeled.png, which
make_composite_figure.py uses for panel B when it exists.
"""
import argparse
import os
import re
import sys

GH = "/scratch/network/sz3962/tau-pet-synthesis-baselines/taugennet"
os.environ["TAUGENNET_ROOT"] = GH
sys.path[:0] = [GH, os.path.join(GH, "scripts")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluate_final as ef  # noqa: E402
from src.dataset_final import build_dataloaders, unnormalize  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help='"<tag>" (dir "single") or "<tag>:fold_<k>"')
    ap.add_argument("--inputs", required=True, help="what the title should say the model used")
    ap.add_argument("--cohort", default="cohort_v2")
    a = ap.parse_args()
    tag, fn = a.run.split(":", 1) if ":" in a.run else (a.run, "single")
    gen_dir = os.path.join(RESULTS, "generated", "baselines", a.cohort, tag, fn)
    rec = os.path.join(RESULTS, "records", "baselines", a.cohort, tag, fn)

    _, _, te, *_ = build_dataloaders("atrophy", use_controls_322=True, use_dk_mask=True)
    rids = [int(re.search(r"RID[_-](\d+)", p).group(1)) for p in te.pet_paths]
    if rids != [int(r) for r in pd.read_csv(os.path.join(gen_dir, "index.csv")).RID]:
        sys.exit("test set order differs from the generated index.csv")
    mask_t = te._dk_mask
    mask = mask_t.squeeze().numpy() > 0

    real, gen = [], []
    with np.load(os.path.join(gen_dir, "predictions.npz")) as z:
        for i, p in enumerate(te.pet_paths):
            v, lo, hi = te._load(p, mask_t)
            real.append(unnormalize(v.squeeze().numpy(), lo, hi))
            gen.append(unnormalize(z[f"subject_{i:03d}"].squeeze(), lo, hi) * mask)

    set1 = pd.read_csv(os.path.join(rec, "pooled_wholebrain_atrophy.csv"), index_col=0)
    r_all = np.concatenate([x[mask] for x in real])
    g_all = np.concatenate([x[mask] for x in gen])
    r = float(np.corrcoef(r_all, g_all)[0, 1])
    if abs(r - float(set1["pearson"].iloc[0])) > 1e-3:
        sys.exit(f"redrawn voxels give R {r:.4f}, scored {float(set1['pearson'].iloc[0]):.4f}")

    out = os.path.join(RESULTS, "figures", "baselines", a.cohort, tag, fn, "pooled_wholebrain_scatter_labeled.png")
    ef.plot_pooled_wholebrain_scatter(real, gen, mask, set1, a.inputs, out)
    print(f"[panel B] {tag}/{fn}: R {r:.4f}, {r_all.size:,} voxels → {out}")


if __name__ == "__main__":
    main()
