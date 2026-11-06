"""
Baseline config. Reuses the TauGenNet lab data pipeline rather than duplicating it:
the sibling taugennet repo is put on sys.path so we can import its dataset loader
(src.dataset_final.build_dataloaders), DK86 mask, and VOL_SHAPE. Evaluation is
delegated to taugennet/scripts/evaluate_final.py (see README) — we do not
reimplement metrics here.
"""

import os
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
TAUGENNET_ROOT = os.environ.get(
    "TAUGENNET_ROOT", "/scratch/network/sz3962/taugennet"
)
if TAUGENNET_ROOT not in sys.path:
    sys.path.insert(0, TAUGENNET_ROOT)

REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "results", "checkpoints")
GENERATED_DIR  = os.path.join(REPO_ROOT, "results", "generated")
LOG_DIR        = os.path.join(REPO_ROOT, "results", "logs")

# ── Re-exported lab-pipeline handles ──────────────────────────────────────────
# Imported lazily-but-eagerly here so callers can `from denseunet.config import ...`.
from src.config import VOL_SHAPE, SEED                     # noqa: E402
from src.dataset_final import build_dataloaders            # noqa: E402

# ── Training defaults (match TauGenNet's split so the baseline is comparable) ──
# build_dataloaders defaults: train_frac=0.64, val_frac=0.16 → 64/16/20 split,
# shuffled with SEED. evaluate_final.py builds its test set the same way, so the
# generated subject_{i:03d}.npy ordering lines up for --use-cached scoring.
LR         = 1e-4
BATCH_SIZE = 2
EPOCHS     = 300
PATIENCE   = 40

# 5-fold CV over the mentor dev pool (test set is the fixed 47 heldout RIDs for
# every fold). build_dataloaders(mode, fold_idx=i, n_folds=N_FOLDS) routes through
# taugennet's mentor split (use_mentor_split defaults True upstream).
N_FOLDS    = 5

__all__ = [
    "TAUGENNET_ROOT", "REPO_ROOT", "CHECKPOINT_DIR", "GENERATED_DIR", "LOG_DIR",
    "VOL_SHAPE", "SEED", "build_dataloaders",
    "LR", "BATCH_SIZE", "EPOCHS", "PATIENCE", "N_FOLDS",
]
