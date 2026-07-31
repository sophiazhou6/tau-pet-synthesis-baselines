#!/usr/bin/env python3
"""
Parse one or more SLURM eval log files and produce a paper-style metrics table.

Usage
-----
# From a single log (all modes in one file, e.g. eval_paper.slurm output):
python scripts/metrics_table.py --log-file results/logs/taugennet_<JOBID>.out

# Explicit per-mode logs:
python scripts/metrics_table.py \
    --log-file results/logs/atrophy.out \
    --log-file results/logs/ptau217.out \
    --log-file results/logs/combined.out

# Write markdown to a file (default: results/records/metrics.md):
python scripts/metrics_table.py --log-file results/logs/eval.out --out results/records/metrics.md

# Print only, don't write:
python scripts/metrics_table.py --log-file results/logs/eval.out --no-write
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── regex patterns ────────────────────────────────────────────────────────────
# Matches evaluate.py and evaluate_v2.py stdout formats:
#   "Mean MAE:   0.0412 ± 0.0089"       (evaluate.py)
#   "Mean MAE  (SUVR):  0.0412 ± 0.0089" (evaluate_v2.py)
_PAT_MAE  = re.compile(r"Mean MAE.*?:\s*([\d.]+)\s*±\s*([\d.]+)")
_PAT_MSE  = re.compile(r"Mean MSE.*?:\s*([\d.]+)\s*±\s*([\d.]+)")
_PAT_SSIM = re.compile(r"Mean SSIM.*?:\s*([\d.]+)\s*±\s*([\d.]+)")
_PAT_PSNR = re.compile(r"Mean PSNR.*?:\s*([\d.]+)\s*±\s*([\d.]+)")

# Mode detection: look for "-- mode atrophy", "Figures → .../atrophy_paper", or
# the evaluate.py/evaluate_v2.py argument line printed by sbatch
_PAT_MODE = re.compile(
    r"(?:--mode\s+|Figures\s+→\s+.*?/)(atrophy|ptau217|combined)",
    re.IGNORECASE,
)

DISPLAY_NAMES = {
    "atrophy":  "Atrophy",
    "ptau217":  "pTau217",
    "combined": "Combined",
}
MODE_ORDER = ["atrophy", "ptau217", "combined"]


def _parse_val(pat, text):
    m = pat.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def parse_log(log_path: str) -> dict[str, dict]:
    """
    Parse a SLURM log file that may contain output from one or more eval runs.
    Returns {mode: {mae, mae_std, mse, mse_std, ssim, ssim_std, psnr, psnr_std}}.
    """
    with open(log_path) as f:
        text = f.read()

    # Split into per-mode blocks by detecting mode boundaries.
    # Insert a sentinel before each mode line so we can split.
    sentinel = "\x00MODE_BOUNDARY\x00"
    tagged = _PAT_MODE.sub(lambda m: sentinel + m.group(0), text)
    blocks = tagged.split(sentinel)

    results = {}
    for block in blocks:
        mode_m = _PAT_MODE.search(block)
        if mode_m is None:
            continue
        mode = mode_m.group(1).lower()

        mae,  mae_std  = _parse_val(_PAT_MAE,  block)
        mse,  mse_std  = _parse_val(_PAT_MSE,  block)
        ssim, ssim_std = _parse_val(_PAT_SSIM, block)
        psnr, psnr_std = _parse_val(_PAT_PSNR, block)

        if mae is None and mse is None and ssim is None:
            continue  # block had no metrics — skip

        entry = results.setdefault(mode, {})
        # Keep first occurrence of each metric (in case multiple blocks match same mode)
        entry.setdefault("mae",      mae)
        entry.setdefault("mae_std",  mae_std)
        entry.setdefault("mse",      mse)
        entry.setdefault("mse_std",  mse_std)
        entry.setdefault("ssim",     ssim)
        entry.setdefault("ssim_std", ssim_std)
        entry.setdefault("psnr",     psnr)
        entry.setdefault("psnr_std", psnr_std)

    return results


def _fmt(val, std, decimals=4):
    if val is None:
        return "—"
    if std is None:
        return f"{val:.{decimals}f}"
    return f"{val:.{decimals}f} ± {std:.{decimals}f}"


def build_markdown(all_results: dict[str, dict], log_files: list[str]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sources = ", ".join(os.path.basename(p) for p in log_files)
    lines = [
        f"## Results (generated {now})",
        f"",
        f"Source log(s): `{sources}`",
        f"",
        f"| Mode | MAE (SUVR) | MSE (SUVR²) | SSIM | PSNR (dB) |",
        f"|------|-----------|------------|------|-----------|",
    ]
    for mode in MODE_ORDER:
        if mode not in all_results:
            continue
        r = all_results[mode]
        name = DISPLAY_NAMES.get(mode, mode.capitalize())
        row = (
            f"| {name} "
            f"| {_fmt(r.get('mae'), r.get('mae_std'))} "
            f"| {_fmt(r.get('mse'), r.get('mse_std'), decimals=6)} "
            f"| {_fmt(r.get('ssim'), r.get('ssim_std'))} "
            f"| {_fmt(r.get('psnr'), r.get('psnr_std'), decimals=2)} |"
        )
        lines.append(row)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Parse SLURM eval logs → markdown metrics table")
    parser.add_argument("--log-file", dest="log_files", action="append", required=True,
                        metavar="PATH", help="SLURM output log (repeat for multiple files)")
    parser.add_argument("--out", default=str(ROOT / "results" / "records" / "metrics.md"),
                        help="Output markdown file (default: results/records/metrics.md)")
    parser.add_argument("--no-write", action="store_true",
                        help="Print table to stdout only, do not write to --out")
    args = parser.parse_args()

    all_results: dict[str, dict] = {}
    for log_path in args.log_files:
        if not os.path.exists(log_path):
            print(f"Warning: log file not found: {log_path}", file=sys.stderr)
            continue
        parsed = parse_log(log_path)
        if not parsed:
            print(f"Warning: no metrics found in {log_path}", file=sys.stderr)
        all_results.update(parsed)

    if not all_results:
        print("Error: no metrics could be extracted from any log file.", file=sys.stderr)
        sys.exit(1)

    md = build_markdown(all_results, args.log_files)

    print(md)

    if not args.no_write:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"Metrics table written to {out_path}", file=sys.stderr)

    return all_results


if __name__ == "__main__":
    main()
