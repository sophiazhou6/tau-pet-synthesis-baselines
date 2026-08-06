#!/usr/bin/env python3
"""
TauGenNet end-to-end experiment orchestrator.

Submits training, polls for completion, runs evaluation, generates a metrics
table, updates README.md, and stages a git commit — all in one recoverable run.

State is persisted to results/experiment_state.json so the script can resume
from the last completed stage after a crash or timeout.

Usage
-----
# Full run (default 70/10/20 split, train_paper.slurm → eval_paper.slurm):
python scripts/run_experiment.py

# Custom split:
python scripts/run_experiment.py --split 80/10/10

# Resume or jump to a specific stage (e.g. after manually running eval):
python scripts/run_experiment.py --from-stage generate_tables \\
    --eval-log results/logs/eval_paper_1234567.out

# Start fresh, ignoring saved state:
python scripts/run_experiment.py --restart
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT       = Path(__file__).parent.parent
STATE_FILE = ROOT / "results" / "experiment_state.json"
PYTHON     = "/home/sz3962/.conda/envs/taugennet/bin/python3"

STAGES = [
    "env_check",
    "submit_training",
    "poll_training",
    "submit_eval",
    "poll_eval",
    "generate_tables",
    "update_readme",
    "prepare_commit",
]

# ── state helpers ─────────────────────────────────────────────────────────────

def _now():
    return datetime.now().isoformat(timespec="seconds")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"run_id": datetime.now().strftime("%Y%m%d_%H%M%S"), "stages": {}}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def is_done(state: dict, stage: str) -> bool:
    return state["stages"].get(stage, {}).get("status") == "done"


def mark_done(state: dict, stage: str, **kwargs):
    state["stages"].setdefault(stage, {}).update(status="done", ts=_now(), **kwargs)
    save_state(state)


def mark_running(state: dict, stage: str, **kwargs):
    state["stages"].setdefault(stage, {}).update(status="running", ts=_now(), **kwargs)
    save_state(state)


def stage_data(state: dict, stage: str) -> dict:
    return state["stages"].get(stage, {})


# ── display helpers ───────────────────────────────────────────────────────────

def _icon(state: dict, stage: str) -> str:
    s = state["stages"].get(stage, {}).get("status")
    return {"done": "✓", "running": "…", "failed": "✗"}.get(s, " ")


def print_todo(state: dict):
    print("\n── Experiment progress ──────────────────────────────────────")
    for s in STAGES:
        icon = _icon(state, s)
        extra = ""
        d = stage_data(state, s)
        if s == "submit_training" and d.get("job_id"):
            extra = f"  (job {d['job_id']})"
        elif s == "submit_eval" and d.get("eval_job_id"):
            extra = f"  (job {d['eval_job_id']})"
        elif s == "poll_training" and d.get("elapsed_s"):
            extra = f"  ({timedelta(seconds=int(d['elapsed_s']))} elapsed)"
        elif s == "generate_tables" and d.get("metrics_md"):
            extra = f"  → {d['metrics_md']}"
        print(f"  [{icon}] {s}{extra}")
    print("─────────────────────────────────────────────────────────────\n")


# ── SLURM helpers ─────────────────────────────────────────────────────────────

def sbatch(script_path: str) -> str:
    """Submit a job and return the numeric job ID."""
    result = subprocess.run(
        ["sbatch", "--parsable", script_path],
        capture_output=True, text=True, check=True,
        cwd=ROOT,
    )
    job_id = result.stdout.strip().split(";")[0]
    return job_id


def job_running(job_id: str) -> bool:
    """Return True if the job is still in the queue (any state)."""
    result = subprocess.run(
        ["squeue", "-j", job_id, "-h"],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def job_state(job_id: str) -> str:
    """Return terminal state from sacct: COMPLETED, FAILED, CANCELLED, etc."""
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
        capture_output=True, text=True,
    )
    states = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    # sacct lists one row per job step; take the first (the batch job itself)
    return states[0].split("+")[0] if states else "UNKNOWN"


def poll_job(job_id: str, label: str, poll_interval: int) -> str:
    """Block until job finishes; return terminal state string."""
    start = time.time()
    dots = 0
    while job_running(job_id):
        elapsed = timedelta(seconds=int(time.time() - start))
        print(f"\r  {label} job {job_id}: running ({elapsed})  {'.' * (dots % 4 + 1)}   ",
              end="", flush=True)
        dots += 1
        time.sleep(poll_interval)
    print()
    return job_state(job_id)


# ── stages ────────────────────────────────────────────────────────────────────

def stage_env_check(state: dict, args):
    print("[env_check] Verifying Python environment …")
    try:
        result = subprocess.run(
            [PYTHON, "-c",
             "import torch, nibabel, skimage; "
             "print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"],
            capture_output=True, text=True, check=True,
        )
        print(f"  {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {e.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    mark_done(state, "env_check", python=PYTHON)


def stage_submit_training(state: dict, args):
    slurm = ROOT / args.slurm
    if not slurm.exists():
        print(f"  ERROR: SLURM script not found: {slurm}", file=sys.stderr)
        sys.exit(1)
    print(f"[submit_training] Submitting {args.slurm} …")
    job_id = sbatch(str(slurm))
    print(f"  Job submitted: {job_id}")
    mark_done(state, "submit_training", job_id=job_id)


def stage_poll_training(state: dict, args):
    job_id = stage_data(state, "submit_training").get("job_id")
    if not job_id:
        print("  ERROR: no training job_id in state — re-run submit_training", file=sys.stderr)
        sys.exit(1)
    print(f"[poll_training] Waiting for job {job_id} …")
    start = time.time()
    terminal = poll_job(job_id, "Training", args.poll_interval)
    elapsed = int(time.time() - start)
    print(f"  Terminal state: {terminal}  (elapsed {timedelta(seconds=elapsed)})")
    if terminal not in ("COMPLETED",):
        print(f"  ERROR: training job ended with state {terminal!r}. "
              "Check the SLURM log before continuing.", file=sys.stderr)
        sys.exit(1)
    mark_done(state, "poll_training", terminal=terminal, elapsed_s=elapsed)


def stage_submit_eval(state: dict, args):
    slurm = ROOT / args.eval_slurm
    if not slurm.exists():
        print(f"  ERROR: eval SLURM script not found: {slurm}", file=sys.stderr)
        sys.exit(1)
    print(f"[submit_eval] Submitting {args.eval_slurm} …")
    job_id = sbatch(str(slurm))

    # Infer log path from the #SBATCH --output line in the script
    log_path = _infer_log_path(slurm, job_id)
    print(f"  Eval job submitted: {job_id}")
    if log_path:
        print(f"  Log will be at: {log_path}")
    mark_done(state, "submit_eval", eval_job_id=job_id, log_path=log_path)


def _infer_log_path(slurm_path: Path, job_id: str) -> str | None:
    """Parse #SBATCH --output from the slurm script and substitute %j → job_id."""
    for line in slurm_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#SBATCH") and "--output" in line:
            val = line.split("=", 1)[-1].strip()
            val = val.replace("%j", job_id)
            return str(ROOT / val)
    return None


def stage_poll_eval(state: dict, args):
    job_id = stage_data(state, "submit_eval").get("eval_job_id")
    if not job_id:
        print("  ERROR: no eval job_id in state — re-run submit_eval", file=sys.stderr)
        sys.exit(1)
    print(f"[poll_eval] Waiting for eval job {job_id} …")
    start = time.time()
    terminal = poll_job(job_id, "Eval", args.poll_interval)
    elapsed = int(time.time() - start)
    print(f"  Terminal state: {terminal}  (elapsed {timedelta(seconds=elapsed)})")
    if terminal not in ("COMPLETED",):
        print(f"  WARNING: eval job ended with state {terminal!r}. "
              "Check the log; tables may be incomplete.")
    mark_done(state, "poll_eval", terminal=terminal, elapsed_s=elapsed)


def stage_generate_tables(state: dict, args):
    print("[generate_tables] Parsing eval log → metrics table …")

    # Prefer explicit --eval-log arg, then state, then glob for latest
    log_path = (args.eval_log
                or stage_data(state, "submit_eval").get("log_path")
                or _latest_eval_log())

    if not log_path or not os.path.exists(log_path):
        print(f"  ERROR: eval log not found ({log_path}). "
              "Pass --eval-log <path> to specify it explicitly.", file=sys.stderr)
        sys.exit(1)

    out_md = ROOT / "results" / "records" / "metrics.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [PYTHON, str(ROOT / "scripts" / "metrics_table.py"),
         "--log-file", log_path,
         "--out", str(out_md)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    mark_done(state, "generate_tables", log_path=log_path, metrics_md=str(out_md))


def _latest_eval_log() -> str | None:
    log_dir = ROOT / "results" / "logs"
    if not log_dir.exists():
        return None
    candidates = sorted(
        (p for p in log_dir.glob("eval_*.out")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def stage_update_readme(state: dict, args):
    print("[update_readme] Writing README.md …")
    metrics_md = stage_data(state, "generate_tables").get("metrics_md",
                             str(ROOT / "results" / "records" / "metrics.md"))
    metrics_text = ""
    if os.path.exists(metrics_md):
        metrics_text = Path(metrics_md).read_text()

    run_id   = state.get("run_id", "unknown")
    split    = args.split

    readme = f"""\
# TauGenNet

3D latent diffusion model for synthesising tau PET brain images conditioned on
structural MRI. Trained on ADNI data (187 subjects, 96×112×96 voxels).

## Architecture

- **Autoencoder**: 3D VAE mapping 96×112×96 PET/MRI to 12×14×12×3 latent space
- **Diffusion U-Net**: denoising network with cross-attention conditioning
- **Conditioning modes**: `atrophy` (86-dim regional z-scores), `ptau217` (scalar plasma),
  `combined` (both)
- **Sampler**: DDPM, 500 inference steps

## Quick Start

```bash
# Full pipeline (AE pretraining + 3 diffusion modes, ~2 days on A100)
sbatch train_paper.slurm

# OR use the orchestrator (auto-polls, evaluates, updates this README)
python scripts/run_experiment.py [--split {split}]
```

See `CLAUDE.md` for the complete canonical pipeline reference.

## Results

{metrics_text.strip() if metrics_text.strip() else "_Run evaluation to populate this table._"}

_Run ID: {run_id} | Split: {split}_

## Repository Layout

```
scripts/
  train.py              main training entry point
  evaluate.py           evaluation suite (all 3 modes)
  run_experiment.py     end-to-end orchestrator (this file's generator)
  metrics_table.py      parse SLURM eval logs → markdown table
  paper_figures.py      publication-quality figures
src/
  config.py             hyperparameters and paths
  dataset_final.py      ADNI dataloader (configurable split)
  dataset_combined.py   combined conditioning dataloader
  models.py             Autoencoder3D + DenoisingUNet3D
  diffusion.py          DDPM noise schedule
  inference.py          synthesis utilities
results/
  checkpoints/          model weights (not version-controlled)
  records/metrics.md    latest evaluation metrics
  figures/              generated figures (not version-controlled)
```
"""
    readme_path = ROOT / "README.md"
    readme_path.write_text(readme)
    print(f"  README.md written ({len(readme)} bytes)")
    mark_done(state, "update_readme", readme=str(readme_path))


def stage_prepare_commit(state: dict, args):
    print("[prepare_commit] Staging files for review …")

    to_stage = [
        "README.md",
        "CLAUDE.md",
        "results/records/metrics.md",
        "src/dataset_final.py",
        "src/dataset_combined.py",
        "scripts/train.py",
        "scripts/run_experiment.py",
        "scripts/metrics_table.py",
    ]
    staged = []
    for path in to_stage:
        full = ROOT / path
        if full.exists():
            subprocess.run(["git", "add", path], cwd=ROOT, check=True)
            staged.append(path)

    print("  Staged files:")
    for p in staged:
        print(f"    {p}")

    result = subprocess.run(
        ["git", "diff", "--cached", "--stat"],
        capture_output=True, text=True, cwd=ROOT,
    )
    print(result.stdout)
    print("  Review with: git diff --cached")
    print("  Commit with: git commit -m 'eval: ...'")
    mark_done(state, "prepare_commit", staged=staged)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TauGenNet experiment orchestrator")
    p.add_argument("--slurm",          default="train_paper.slurm",
                   help="Training SLURM script (relative to repo root)")
    p.add_argument("--eval-slurm",     default="eval_paper.slurm",
                   help="Evaluation SLURM script (relative to repo root)")
    p.add_argument("--split",          default="70/10/20",
                   help="Data split for documentation purposes (does not modify SLURM scripts)")
    p.add_argument("--poll-interval",  type=int, default=60,
                   help="Seconds between squeue polls (default 60)")
    p.add_argument("--restart",        action="store_true",
                   help="Ignore saved state and re-run all stages from scratch")
    p.add_argument("--from-stage",     choices=STAGES, default=None,
                   help="Reset state from this stage onward and resume from here")
    p.add_argument("--eval-log",       default=None,
                   help="Path to SLURM eval log (used by generate_tables if set)")
    return p.parse_args()


def main():
    args = parse_args()

    state = load_state() if not args.restart else {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "stages": {},
    }

    if args.from_stage:
        idx = STAGES.index(args.from_stage)
        for s in STAGES[idx:]:
            state["stages"].pop(s, None)
        save_state(state)

    print_todo(state)

    dispatch = {
        "env_check":       stage_env_check,
        "submit_training": stage_submit_training,
        "poll_training":   stage_poll_training,
        "submit_eval":     stage_submit_eval,
        "poll_eval":       stage_poll_eval,
        "generate_tables": stage_generate_tables,
        "update_readme":   stage_update_readme,
        "prepare_commit":  stage_prepare_commit,
    }

    for stage in STAGES:
        if is_done(state, stage):
            continue
        mark_running(state, stage)
        try:
            dispatch[stage](state, args)
        except SystemExit:
            state["stages"][stage]["status"] = "failed"
            save_state(state)
            print_todo(state)
            raise
        print_todo(state)

    print("All stages complete. Review staged files with: git diff --cached")


if __name__ == "__main__":
    main()
