"""Compact metrics over immutable decisions and continuous anomaly scores.

Uses the frozen metric implementations without changing their public contract.
Range F1 uses flat positional bias and alpha=0.2 existence reward.
AUC PR is trapezoidal area under the precision-recall curve.
"""

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.f1_score_f1_pa import get_adjust_F1PA
from metrics.vus.metrics import get_range_vus_roc
from metrics.vus.utils.metrics import metricor


def harmonic(precision, recall):
    return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def decision_metrics(scores, labels, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    pred = (np.asarray(scores, dtype=np.float64) > threshold).astype(np.int64)
    events = convert_vector_to_events(labels)
    grader = metricor()
    range_recall = grader.range_recall_new(labels, pred, 0.2)[0]
    range_precision = grader.range_recall_new(pred, labels, 0)[0]
    affiliation = pr_from_events(convert_vector_to_events(pred), events, (0, len(labels))) if events else None
    affiliation_f1 = None
    if affiliation is not None:
        precision, recall = affiliation["precision"], affiliation["recall"]
        # Empty predictions give NaN affiliation precision in the frozen code.
        affiliation_f1 = harmonic(precision, recall) if np.isfinite(precision) else 0.0
    result = {
        "threshold": float(threshold),
        "event_f1": harmonic(
            float(np.sum(pred * labels) / max(np.sum(pred), 1)),
            float(np.mean([pred[start:end].any() for start, end in events])) if events else 0.0),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "range_f1": harmonic(range_precision, range_recall),
        "affiliation_f1": affiliation_f1,
    }
    # The frozen PA routine mutates its input. Keep all other metrics and
    # the caller's predictions untouched, and compute PA only at the end.
    # A leading normal sentinel also handles events starting at index zero
    # despite the frozen PA implementation's backward loop excluding zero.
    result["pa_f1"] = (float(get_adjust_F1PA(np.pad(pred, (1, 0)), np.pad(labels, (1, 0)))[3]) if events
                       else float(f1_score(labels, pred, zero_division=0)))
    return result


def score_metrics(scores, labels, vus_window):
    scores, labels = np.asarray(scores), np.asarray(labels)
    if np.unique(labels).size < 2:
        return dict(vus_pr=None, vus_roc=None, auc_pr=None, auc_roc=None)
    precision, recall, _ = precision_recall_curve(labels, scores)
    vus = get_range_vus_roc(scores, labels, vus_window)
    return {"vus_pr": float(vus["VUS_PR"]), "vus_roc": float(vus["VUS_ROC"]),
            "auc_pr": float(auc(recall, precision)),
            "auc_roc": float(roc_auc_score(labels, scores))}
