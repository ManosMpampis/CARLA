"""Run with ./venv/bin/python tests/check_evaluation_scores.py."""

from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.metrics import combine_all_evaluation_scores
from metrics.vus.metrics import get_range_vus_roc
from metrics.Matthews_correlation_coefficient import MCC
from metrics.f1_score_f1_pa import event_f1, get_adjust_F1PA


scores = np.array([0.1, 0.2, 0.7, 0.4, 0.8, 0.3, 0.6, 0.9, 0.2, 0.1])
labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32)
original_scores, original_labels = scores.copy(), labels.copy()
expected_vus = get_range_vus_roc(scores, labels, 2)

for threshold in (0.7, 0.8):
    result = combine_all_evaluation_scores(scores, labels, threshold, window_size=2)
    predictions = (scores >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (tn, fp, fn, tp)
    assert np.isclose(result["f1_score"], f1_score(labels, predictions))

    # Each event contains a detection at both thresholds; PA fills both events.
    adjusted_predictions = predictions.copy()
    adjusted_predictions[2:5] = 1
    adjusted_predictions[7:9] = 1
    assert np.isclose(result["pa_f1_score"], f1_score(labels, adjusted_predictions))

    # Changing a decision threshold or applying PA must not change score metrics.
    for key, expected in expected_vus.items():
        assert np.isclose(result[key], expected), key
    np.testing.assert_array_equal(scores, original_scores)
    np.testing.assert_array_equal(labels, original_labels)

print("Continuous score metrics, >= thresholds, PA decisions, and input preservation passed.")

# Events have exclusive ends; the detection immediately after an event must
# not count as detecting it. A second correctly detected event keeps precision
# nonzero so an erroneous event recall is observable.
assert np.isclose(event_f1(np.array([1, 0, 0, 1, 0]), [(0, 1), (3, 4)],
                          np.array([0, 1, 0, 1, 0])), 0.5)
pred = np.array([0, 1, 0])
assert get_adjust_F1PA(pred, np.array([1, 1, 0]))[3] == 1.0
np.testing.assert_array_equal(pred, [1, 1, 0])

for labels in (np.zeros(4, dtype=int), np.ones(4, dtype=int)):
    result = combine_all_evaluation_scores(np.array([0.1, 0.2, 0.8, 0.9]), labels, 0.5, 1)
    for key in ("R_AUC_ROC", "R_AUC_PR", "VUS_ROC", "VUS_PR"):
        assert result[key] is None
    assert np.isfinite(result["pa_latency"])

# The MCC denominator used to overflow int64 for long timeseries.
labels = np.tile([0, 0, 1, 1], 100_000)
predictions = np.tile([0, 1, 1, 1], 100_000)
assert np.isclose(MCC(labels, predictions), matthews_corrcoef(labels, predictions))
print("Exclusive event ends, leading PA events, single-class labels, and long-series MCC passed.")
