"""ORACLE calibration probe (diagnostic only, NOT honest evaluation).

Sweeps decision thresholds directly on the TEST labels of a finished score
run to answer "what is the best this score map could possibly do?". The
numbers it prints are an upper bound for threshold tuning -- they must
never be compared against honest metrics (clean-train-only thresholds in
metrics.json) or published as results.

Usage:
    ./venv/bin/python oracle_threshold.py \
        results/smd/lewm_true_demo_score/machine-1-1.txt/jepa/scores.npz
"""
import sys

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)


def sweep(scores, labels, n_grid=2000):
    """Best point-F1 / MCC over a threshold grid spanning the score range."""
    lo, hi = float(scores.min()), float(scores.max())
    grid = np.unique(np.quantile(scores, np.linspace(0, 1, n_grid)))
    grid = np.concatenate([[lo - 1e-9], grid])
    best_f1, best_mcc = None, None
    for thr in grid:
        pred = (scores >= thr).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        f1 = float(f1_score(labels, pred, zero_division=0))
        mcc = float(matthews_corrcoef(labels, pred))
        if best_f1 is None or f1 > best_f1[0]:
            best_f1 = (f1, float(thr), float(precision_score(labels, pred, zero_division=0)),
                       float(recall_score(labels, pred, zero_division=0)))
        if best_mcc is None or mcc > best_mcc[0]:
            best_mcc = (mcc, float(thr))
    return best_f1, best_mcc


def main(path):
    d = np.load(path)
    labels = d["gt_labels"].astype(int)
    print(f"run: {path}  n={len(labels)} "
          f"anomaly_rate={labels.mean():.4f}\n")
    print("NOTE: oracle thresholds are fit on TEST labels -- ceiling only.\n")
    for key in ["scores"] + sorted(k for k in d.files if k.startswith("channel/")):
        s = np.asarray(d[key], dtype=np.float64)
        (f1, thr_f1, prec, rec), (mcc, thr_mcc) = sweep(s, labels)
        print(f"{key:14s} AUROC={roc_auc_score(labels, s):.4f} "
              f"AP={average_precision_score(labels, s):.4f} | "
              f"best-F1={f1:.4f} @thr={thr_f1:.4g} (P={prec:.4f} R={rec:.4f}) | "
              f"best-MCC={mcc:.4f} @thr={thr_mcc:.4g}")


if __name__ == "__main__":
    main(sys.argv[1])
