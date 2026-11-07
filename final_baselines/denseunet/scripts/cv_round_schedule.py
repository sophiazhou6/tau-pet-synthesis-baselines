#!/home/sz3962/.conda/envs/taugennet/bin/python3
"""
Training length for the full-pool refit of an upstream_rounds model (denseunet_paper), from its
5-fold CV: for every round, where that round's best validation epoch fell in each fold (1 = its
first epoch), and the median over folds. The median schedule is what the refit replays on the
whole development pool without a validation set (--val-frac 0 --round-schedule ...).

    python scripts/cv_round_schedule.py --cohort cohort_v2 --tag denseunet_paper
Writes round_schedule.json next to the fold checkpoints and prints the --round-schedule value.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denseunet.config import CKPT_ROOT  # noqa: E402


def best_epoch_per_round(history):
    """{round: 1-based position of its lowest val_mse} — the first minimum, as the trainer keeps
    the first strict improvement."""
    out = {}
    for r in sorted({h["round"] for h in history}):
        vals = [h["val_mse"] for h in history if h["round"] == r]
        out[r] = int(np.argmin(vals)) + 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    a = p.parse_args()
    base = os.path.join(CKPT_ROOT, a.cohort, a.tag)
    per_fold = {}
    for f in a.folds:
        path = os.path.join(base, f"fold_{f}", "history.json")
        if not os.path.exists(path):
            sys.exit(f"missing {path} — is fold {f} finished?")
        per_fold[f] = best_epoch_per_round(json.load(open(path)))
    rounds = sorted(set.intersection(*(set(v) for v in per_fold.values())))
    if any(set(v) != set(rounds) for v in per_fold.values()):
        sys.exit(f"folds ran different rounds: { {f: sorted(v) for f, v in per_fold.items()} }")
    schedule = [int(np.median([per_fold[f][r] for f in a.folds])) for r in rounds]

    print("best epoch within each round (1 = first epoch of the round)")
    print("round  " + "  ".join(f"f{f:<2d}" for f in a.folds) + "  median")
    for i, r in enumerate(rounds):
        print(f"{r:5d}  " + "  ".join(f"{per_fold[f][r]:3d}" for f in a.folds) + f"  {schedule[i]:6d}")
    print(f"total {sum(schedule)} epochs; fold totals to their round bests: "
          f"{ {f: sum(v.values()) for f, v in per_fold.items()} }")
    out = {"tag": a.tag, "folds": a.folds, "per_fold": {str(f): v for f, v in per_fold.items()},
           "schedule": schedule}
    with open(os.path.join(base, "round_schedule.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"--round-schedule {','.join(map(str, schedule))}")


if __name__ == "__main__":
    main()
