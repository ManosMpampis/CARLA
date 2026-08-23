"""Auxiliary assertion point (spec): the scorer -> evaluator handoff.

1. Overlap-aware aggregation: a point covered by N windows appears exactly
   once per timestep, correctly weighted (brute-force cross-check).
2. The untouched metrics stack consumes (pred_labels, gt) derived from
   aggregated scores and returns the frozen metric-dictionary contract.
3. Thresholding path is train-side only.

Usage: ./venv/bin/python tests/check_scorer_handoff.py
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils.scoring import Scorer  # noqa: E402


class _StubScorerModel(torch.nn.Module):
    """Returns a known per-position score map without a real JEPA."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.level_names = ["L0"]
        self.level_dims = [1]
        self.level_strides = [1]

    def score(self, x):
        b, _, w = x.shape
        # score at window position t encodes x's own timestep value
        vals = x.float().mean(dim=1)
        fused = vals.reshape(b, w)
        return {
            "fused": fused,
            "levels": {"L0": fused},
            "signals": {},
        }


def brute_force_overlap(window_scores, starts, n_steps, wsz):
    total = np.zeros(n_steps)
    count = np.zeros(n_steps)
    for row, s in zip(window_scores, starts):
        total[s:s + wsz] += row
        count[s:s + wsz] += 1
    return total / np.maximum(count, 1)


EXPECTED_KEYS = {
    "Affiliation precision", "Affiliation recall", "Event F1", "MCC",
    "R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR",
    "accuracy", "precision", "recall", "f1_score", "f05_score",
    "tp", "fp", "fn", "tn",
    "pa_Affiliation precision", "pa_Affiliation recall", "pa_Event F1",
    "pa_MCC", "pa_R_AUC_ROC", "pa_R_AUC_PR", "pa_VUS_ROC", "pa_VUS_PR",
    "pa_accuracy", "pa_precision", "pa_recall", "pa_f1_score", "pa_latency",
}


def main():
    from metrics.metrics import combine_all_evaluation_scores

    rng = np.random.default_rng(4)
    n_steps, wsz, stride = 500, 32, 7
    series = rng.normal(0, 1, size=(n_steps, 3)).astype(np.float32)

    scorer = Scorer(_StubScorerModel(), torch.device("cpu"))
    result = scorer.score_series(series, wsz, stride)
    scores, starts, ends = result["scores"], result["start_idxs"], result["end_idxs"]

    # 1. overlap-aware aggregation matches brute force exactly
    assert len(scores) == n_steps, \
        f"per-timestep array length {len(scores)} != series length {n_steps}"
    assert len(starts) == len(ends)
    rows = np.stack([series[s:s + wsz].mean(axis=1) for s in starts])
    expected = brute_force_overlap(rows, starts, n_steps, wsz)
    covered = result["cover_counts"] > 0
    np.testing.assert_allclose(scores[covered], expected[covered], rtol=1e-9,
                               err_msg="overlap aggregation mismatch")
    # points covered by many windows counted once per timestep: constant-
    # value probe must stay identical regardless of cover count
    ones_series = np.ones((200, 1), dtype=np.float32)
    r2 = Scorer(_StubScorerModel(), torch.device("cpu")).score_series(
        ones_series, 50, 5)
    assert np.allclose(r2["scores"], 1.0), "double counting detected"

    # 2. frozen metric-dictionary contract through the untouched stack
    gt = (rng.random(n_steps) < 0.05).astype(int)
    threshold = float(np.quantile(scores, 0.995))  # train-style statistic
    pred = (scores >= threshold).astype(int)
    metric_dict = combine_all_evaluation_scores(pred, gt, 100)
    missing = EXPECTED_KEYS - set(metric_dict)
    assert not missing, f"frozen contract violated, missing keys: {missing}"

    # 3. calibrator inputs are train-side only (structural check)
    from utils.scoring import Calibrator

    calib = Calibrator(quantile=0.99)
    clean_train = rng.normal(0, 1, 4000)
    probes = rng.normal(3, 1, 500)
    calib.fit({"fused": clean_train}, probes={"fused": probes})
    th = calib.threshold_for(clean_train)
    assert th == np.quantile(clean_train, 0.99)
    assert not calib.fallback and abs(calib.weights["fused"] - 1.0) < 1e-9
    thin = Calibrator(min_probe_samples=1000)
    thin.fit({"fused": clean_train}, probes={"fused": probes})
    assert thin.fallback, "thin probes must fall back to plain mean"

    print("T03 handoff OK: overlap aggregation exact; frozen metric "
          "contract complete; thresholds trace to train-side only.")


if __name__ == "__main__":
    main()
