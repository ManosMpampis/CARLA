"""Verify saved replay/score agreement without loading a model or rescoring data."""

import base64
import json
from pathlib import Path
import re

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def verify(machine):
    directory = ROOT / "results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features" / machine
    candidates = list(directory.glob("jepa*"))
    if len(candidates) != 1:
        raise ValueError(f"ambiguous artifact directory for {machine}")
    directory = candidates[0]
    trace = json.loads((directory / "decision_trace.json").read_text())
    report = json.loads((directory / "metrics.json").read_text())
    html = (directory / "decision_trace.html").read_text()
    encoded = re.search(r"const encodedTrace = '([^']+)'", html).group(1)
    embedded = json.loads(base64.b64decode(encoded))
    with np.load(directory / "scores.npz") as arrays:
        size = len(arrays["timeseries_scores"])
        final = np.full(size, np.nan)
        counts = np.zeros(size, dtype=int)
        decisions = np.zeros(size, dtype=bool)
        recovered = []
        indices_valid, counts_valid = True, True
        all_decisions_valid = True
        for update in trace["updates"]:
            ix = np.asarray(update["indices"])
            values = np.asarray(update["scores"])
            current_counts = np.asarray(update["counts"])
            previous = np.nan_to_num(final[ix])
            # A stored update is a RUNNING mean, not a new raw score map.
            raw = current_counts * values - (current_counts - 1) * previous
            recovered.append(float(raw.mean()))
            indices_valid &= np.array_equal(ix, np.arange(update["checking_start"], update["checking_end"]))
            counts_valid &= np.array_equal(current_counts, counts[ix] + 1)
            decisions[ix] = update["decisions"]
            expected_decisions = values > trace["threshold"] if trace["threshold_operator"] == "gt" else values >= trace["threshold"]
            all_decisions_valid &= np.array_equal(decisions[ix], expected_decisions)
            final[ix] = values
            counts[ix] = current_counts
        covered = counts > 0
        indices = np.flatnonzero(covered)
        truth = np.zeros(size, dtype=bool)
        for start, end in trace["ground_truth_intervals"]:
            truth[start:end] = True
        saved = arrays["timeseries_scores"][covered]
        replay = final[covered]
        labels = arrays["timestep_labels"][covered]
        differences = np.abs(saved - replay)
        relative = differences / np.maximum(np.abs(saved), 1e-12)
        worst = int(np.argmax(differences))
        changed = np.flatnonzero((saved > trace["threshold"]) != (replay > trace["threshold"]))
        window_scores = np.asarray(recovered)
        window_threshold = report["evaluation"]["window"]["calibrated"]["threshold"]
        row = dict(
            machine=machine, n_updates=len(trace["updates"]), html_json_equal=trace == embedded,
            scope=trace["scope"], support=[int(indices[0]), int(indices[-1]) + 1],
            index_contract=bool(indices_valid), counts_increment_contract=bool(counts_valid),
            counts_vs_npz=bool(np.array_equal(counts, arrays["cover_counts"])),
            all_trace_decisions_match_operator=bool(all_decisions_valid),
            threshold=trace["threshold"], threshold_operator=trace["threshold_operator"],
            timeseries_threshold=report["evaluation"]["timeseries"]["calibrated"]["threshold"],
            window_threshold=window_threshold,
            absdiff_quantiles_p50_p90_p99_p999_max=np.quantile(differences, [.5, .9, .99, .999, 1]).tolist(),
            reldiff_quantiles_p50_p90_p99_p999_max=np.quantile(relative, [.5, .9, .99, .999, 1]).tolist(),
            final_score_max_abs_diff=float(differences.max()),
            final_score_rmse=float(np.sqrt(np.mean(differences ** 2))),
            maxdiff_point=dict(index=int(indices[worst]), saved=float(saved[worst]),
                               replay=float(replay[worst]), relative=float(relative[worst])),
            max_recovered_window_score_diff=float(np.max(np.abs(window_scores - arrays["window_scores"]))),
            changed_decisions=[dict(index=int(indices[i]), label=int(labels[i]),
                                    saved=float(saved[i]), replay=float(replay[i])) for i in changed],
            truth_matches_npz=bool(np.array_equal(truth, arrays["timestep_labels"] != 0)),
            window_labels_match_npz=bool(np.array_equal(arrays["window_labels"],
                [np.any(truth[s:e]) for s, e in zip(arrays["start_idxs"], arrays["end_idxs"])])),
            input_starts_match_npz=bool(np.array_equal([u["window_start"] for u in trace["updates"]], arrays["input_start_idxs"])),
            output_starts_match_npz=bool(np.array_equal([u["checking_start"] for u in trace["updates"]], arrays["start_idxs"])),
            output_ends_match_npz=bool(np.array_equal([u["checking_end"] for u in trace["updates"]], arrays["end_idxs"])),
            npz_timeseries_cal_f1=float(f1_score(labels, saved > trace["threshold"])),
            replay_timeseries_cal_f1=float(f1_score(labels, replay > trace["threshold"])),
            stored_timeseries_cal_f1=report["evaluation"]["timeseries"]["calibrated"]["f1_no_pa"],
            recovered_window_cal_f1=float(f1_score(arrays["window_labels"], window_scores > window_threshold)),
            npz_window_cal_f1=float(f1_score(arrays["window_labels"], arrays["window_scores"] > window_threshold)),
            recovered_window_prediction_mismatch=int(np.sum((window_scores > window_threshold) != (arrays["window_scores"] > window_threshold))),
        )
    return row


def main():
    rows = [verify(machine) for machine in ("machine-1-1.txt", "machine-2-3.txt", "machine-3-3.txt")]
    (OUT / "trace_sample_verification.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(json.dumps({k: row[k] for k in ("machine", "html_json_equal", "counts_vs_npz", "truth_matches_npz", "final_score_max_abs_diff", "npz_timeseries_cal_f1", "replay_timeseries_cal_f1", "recovered_window_prediction_mismatch")}))


if __name__ == "__main__":
    main()
