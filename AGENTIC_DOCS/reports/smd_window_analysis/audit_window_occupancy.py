"""Read-only descriptive audit of selected saved SMD window score archives.

Run from the repository root:
    ./venv/bin/python reports/smd_window_analysis/audit_window_occupancy.py

Uses the actual scored target span in start_idxs/end_idxs, not the whole input
span. Never reads timeseries or non-overlap scores; timestep labels only recover
the fraction of an existing scored window occupied by annotated anomalies.
"""

from pathlib import Path
import csv
import json

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
FAMILIES = {
    "frequency_phase1": "lewm_encoder/frequency/phase1",
    "frequency_aux_phase1": "lewm_encoder/frequency_aux/phase1",
    "ae": "ae/default",
    "frequency_aux_observed_query": "lewm_encoder/frequency_aux/cross_attention/observed_query",
}


def average(values):
    values = [float(x) for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(values)) if values else None


def correlation(a, b, kind):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b).statistic if kind == "spearman"
                 else np.corrcoef(a, b)[0, 1])


def audit(family, path):
    metrics = json.loads((path.parent / "metrics.json").read_text())
    window = metrics["evaluation"]["window"]
    with np.load(path, allow_pickle=False) as archive:
        score = archive["window_scores"].astype(float)
        label = archive["window_labels"].astype(bool)
        starts = archive["start_idxs"].astype(int)
        ends = archive["end_idxs"].astype(int)
        timeline = archive["timestep_labels"].astype(bool)
        saved_pred = archive["window_predictions"].astype(bool)
    assert len(score) == len(label) == len(starts) == len(ends)
    assert np.isfinite(score).all()
    assert ((starts >= 0) & (ends > starts) & (ends <= len(timeline))).all()
    prefix = np.r_[0, np.cumsum(timeline, dtype=np.int64)]
    occupancy = (prefix[ends] - prefix[starts]) / (ends - starts)
    assert np.array_equal(occupancy > 0, label), f"window label/span mismatch: {path}"
    pred = score > float(window["calibrated"]["threshold"])
    assert np.array_equal(pred, saved_pred), f"saved prediction mismatch: {path}"
    positive = label
    false_negative = positive & ~pred
    true_positive = positive & pred
    low = positive & (occupancy <= .10)
    high = occupancy > .50
    prevalence = float(label.mean())
    baseline = 2 * prevalence / (1 + prevalence)
    oracle = float(window["oracle"]["f1_no_pa"])
    machine = next(p.name for p in path.parents if p.name.startswith("machine-"))
    return {
        "family": family, "machine": machine,
        "scores_path": str(path.relative_to(ROOT)),
        "window_count": len(label), "positive_count": int(positive.sum()),
        "scored_span_min": int(np.min(ends-starts)),
        "scored_span_max": int(np.max(ends-starts)),
        "positive_prevalence": prevalence,
        "always_positive_f1": baseline, "oracle_f1": oracle,
        "oracle_above_always_positive": oracle - baseline,
        "oracle_equals_always_positive": bool(np.isclose(oracle, baseline, atol=1e-10, rtol=0)),
        "score_occupancy_spearman_positive": correlation(score[positive], occupancy[positive], "spearman"),
        "score_occupancy_pearson_positive": correlation(score[positive], occupancy[positive], "pearson"),
        "score_occupancy_spearman_all": correlation(score, occupancy, "spearman"),
        "mean_occupancy_positive": average(occupancy[positive]),
        "false_negative_count": int(false_negative.sum()),
        "true_positive_count": int(true_positive.sum()),
        "mean_occupancy_false_negative": average(occupancy[false_negative]),
        "mean_occupancy_true_positive": average(occupancy[true_positive]),
        "recall_low_occupancy": average(pred[low]),
        "low_occupancy_count": int(low.sum()),
        "recall_high_occupancy": average(pred[high]),
        "high_occupancy_count": int(high.sum()),
        "calibrated_positive_recall": average(pred[positive]),
    }


def main():
    rows = []
    for family, relative in FAMILIES.items():
        paths = sorted((ROOT / "results/smd" / relative).glob("machine-*/*/scores.npz"))
        assert len(paths) == 28, (family, len(paths))
        rows.extend(audit(family, path) for path in paths)
    summary = {}
    for family in FAMILIES:
        group = [r for r in rows if r["family"] == family]
        equal = [r["machine"] for r in group if r["oracle_equals_always_positive"]]
        values = {"machine_count": len(group), "oracle_equals_always_positive_machines": equal}
        fields = ["score_occupancy_spearman_positive", "score_occupancy_pearson_positive",
                  "score_occupancy_spearman_all", "mean_occupancy_positive",
                  "mean_occupancy_false_negative", "mean_occupancy_true_positive",
                  "recall_low_occupancy", "recall_high_occupancy", "calibrated_positive_recall",
                  "positive_prevalence", "always_positive_f1", "oracle_above_always_positive"]
        for field in fields:
            eligible = [r[field] for r in group if r[field] is not None]
            values[field + "_macro_mean"] = average(eligible)
            values[field + "_eligible_machines"] = len(eligible)
        paired = [r for r in group if r["mean_occupancy_false_negative"] is not None
                  and r["mean_occupancy_true_positive"] is not None]
        values["fn_minus_tp_occupancy_paired_macro_mean"] = average(
            [r["mean_occupancy_false_negative"] - r["mean_occupancy_true_positive"] for r in paired])
        values["fn_tp_paired_machine_count"] = len(paired)
        values["fn_lower_occupancy_than_tp_machine_count"] = sum(
            r["mean_occupancy_false_negative"] < r["mean_occupancy_true_positive"] for r in paired)
        recall_paired = [r for r in group if r["recall_low_occupancy"] is not None
                         and r["recall_high_occupancy"] is not None]
        values["recall_low_high_paired_machine_count"] = len(recall_paired)
        values["recall_low_occupancy_paired_macro_mean"] = average(
            [r["recall_low_occupancy"] for r in recall_paired])
        values["recall_high_occupancy_paired_macro_mean"] = average(
            [r["recall_high_occupancy"] for r in recall_paired])
        values["recall_high_minus_low_paired_macro_mean"] = average(
            [r["recall_high_occupancy"] - r["recall_low_occupancy"] for r in recall_paired])
        summary[family] = values
    (OUTPUT / "window_occupancy.json").write_text(json.dumps({
        "method": "Saved validation-selected base-run score archives; per-machine calculations then equal-machine means among eligible machines. Low occupancy: 0<f<=0.1; high: f>0.5. Positive-only correlations exclude normal windows. Threshold: saved calibrated threshold with strict greater-than. No model inference or training.",
        "summary": summary, "machines": rows}, indent=2) + "\n")
    with (OUTPUT / "window_occupancy.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
