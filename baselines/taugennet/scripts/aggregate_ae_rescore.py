#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
aggregate_ae_rescore.py — collect the DK86-rescore AE recon JSONs into one markdown table.

The rescore_ae_dk86 job writes one JSON per AE (latent_ch × lp_weight ablation + the lch8
baseline) via eval_ae_recon.py --metrics-out, each holding subject-mean in-DK86 metrics:
    {"mse", "mae", "nrmse", "pearson", "ssim", "n"}

This reader globs those JSONs and emits a sorted markdown table — nothing is recomputed.

Usage:
  python scripts/aggregate_ae_rescore.py
  python scripts/aggregate_ae_rescore.py --json-dir results/records/ae_rescore_dk86 --force
"""
import argparse
import glob
import json
import os

METRIC_KEYS = ["ssim", "pearson", "nrmse", "mse", "mae"]
ARROW = {"ssim": "↑", "pearson": "↑", "nrmse": "↓", "mse": "↓", "mae": "↓"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json-dir", default="results/records/ae_rescore_dk86")
    p.add_argument("--out", default=None, help="Default: <json-dir>/summary.md")
    p.add_argument("--force", action="store_true", help="Overwrite an existing summary.")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    if not paths:
        raise SystemExit(f"no *.json found in {args.json_dir} (has rescore_ae_dk86 finished?)")

    rows = []
    for path in paths:
        tag = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            d = json.load(f)
        rows.append((tag, d))

    # Stable, readable ordering: by latent_ch then lp_weight, baseline last.
    def sort_key(item):
        tag = item[0]
        return (0 if tag.startswith("lch4") else 1 if tag.startswith("lch8") else 2,
                "baseline" in tag, tag)
    rows.sort(key=sort_key)

    hdr = "| AE | n | " + " | ".join(f"{k.upper()} {ARROW[k]}" for k in METRIC_KEYS) + " |"
    sep = "|" + "---|" * (2 + len(METRIC_KEYS))
    lines = ["# AE reconstruction — DK86-masked rescore (held-out test split)\n",
             "Subject-mean in-DK86 metrics from eval_ae_recon.py.\n", hdr, sep]
    for tag, d in rows:
        n = d.get("n", "")
        cells = [f"{d[k]:.4f}" if k in d else "–" for k in METRIC_KEYS]
        lines.append(f"| {tag} | {n} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + "\n"

    out = args.out or os.path.join(args.json_dir, "summary.md")
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"refusing to overwrite {out} (pass --force)")
    with open(out, "w") as f:
        f.write(table)
    print(table)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
