"""
Paths and shared constants. Mirrors denseunet/config.py: the TauGenNet repo is put on
sys.path so its dataset class / split policy / metric functions can be imported (never
copied), keeping every baseline on the exact same data pipeline and scorer.
"""
import os
import sys

TAUGENNET_ROOT = os.environ.get("TAUGENNET_ROOT", "/scratch/network/sz3962/taugennet")
for _p in (TAUGENNET_ROOT, os.path.join(TAUGENNET_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR  = os.path.join(REPO_ROOT, "results")
CKPT_ROOT    = os.path.join(RESULTS_DIR, "checkpoints", "baselines")
GEN_ROOT     = os.path.join(RESULTS_DIR, "generated", "baselines")
RECORDS_ROOT = os.path.join(RESULTS_DIR, "records", "baselines")
FIGURES_ROOT = os.path.join(RESULTS_DIR, "figures", "baselines")
LOG_DIR      = os.path.join(RESULTS_DIR, "logs")

# Cohort manifests live next to the raw data (one source of truth for the data), not in
# the code repo. Override with BASELINES_MANIFEST_DIR.
MANIFEST_DIR = os.environ.get(
    "BASELINES_MANIFEST_DIR", os.path.join(TAUGENNET_ROOT, "data", "manifests"))

# The user's GitHub repo (sophiazhou6/tau-pet-synthesis-baselines) checked out on scratch: its
# taugennet/scripts/evaluate_final.py + localization_metrics.py ARE the evaluation pipeline every
# baseline is scored with (scripts/score_baseline.py --pipeline github, the default).
GH_ROOT = os.environ.get("TAU_GH_ROOT", "/scratch/network/sz3962/tau-pet-synthesis-baselines")
GH_TAUGENNET = os.path.join(GH_ROOT, "taugennet")

from src.config import VOL_SHAPE, SEED  # noqa: E402  (lab volume shape (96,112,96), seed 42)

N_FOLDS = 5

__all__ = ["TAUGENNET_ROOT", "REPO_ROOT", "RESULTS_DIR", "CKPT_ROOT", "GEN_ROOT",
           "RECORDS_ROOT", "FIGURES_ROOT", "LOG_DIR", "MANIFEST_DIR", "GH_ROOT", "GH_TAUGENNET",
           "VOL_SHAPE", "SEED", "N_FOLDS"]
