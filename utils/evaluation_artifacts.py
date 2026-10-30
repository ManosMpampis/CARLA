"""Stream full scores to CSV and render bounded interactive score reports."""

import base64
import csv
from pathlib import Path

import numpy as np

from utils.evaluation_logging import MODES, evaluation_row


def write_score_html(path, scores, labels, threshold, title, *, timestep_indices=None,
                     series_labels=None, window_starts=None, window_ends=None,
                     input_starts=None, input_ends=None):
    """Keep every score once; render a pixel-bounded view and real checking windows."""
    from utils.decision_trace import save_decision_trace_html

    scores, labels = np.asarray(scores), np.asarray(labels)
    timeline_labels = np.asarray(labels if series_labels is None else series_labels)
    indices = np.arange(len(scores), dtype=np.float64) if timestep_indices is None else np.asarray(timestep_indices, dtype=np.float64)
    if not len(scores) or len(scores) != len(indices) or not np.isfinite(scores).all():
        raise ValueError("HTML scores require one finite score per timestep index")
    if np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= len(timeline_labels):
        raise ValueError("HTML timestep indices must increase within the original series")
    window_starts = [0] if window_starts is None else window_starts
    window_ends = [len(timeline_labels)] if window_ends is None else window_ends
    input_starts = window_starts if input_starts is None else input_starts
    input_ends = window_ends if input_ends is None else input_ends
    lengths = {len(values) for values in (window_starts, window_ends, input_starts, input_ends)}
    if len(lengths) != 1 or not len(window_starts):
        raise ValueError("HTML checking/input window arrays must have equal nonzero lengths")
    def packed(values, dtype):
        return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")
    active = (timeline_labels != 0).astype(np.int8)
    edges = np.diff(np.pad(active, (1, 1)))
    intervals = list(zip(np.flatnonzero(edges == 1).tolist(),
                         np.flatnonzero(edges == -1).tolist()))
    simplified = len(intervals) > 2048
    if simplified:
        # Bound SVG bands too; preserve every anomaly, marking merged spans.
        intervals = [(intervals[int(chunk[0])][0], intervals[int(chunk[-1])][1])
                     for chunk in np.array_split(np.arange(len(intervals)), 2048)]
    trace = {
        "final_view": True, "title": title, "score_label": "Anomaly score",
        "threshold": float(threshold), "series_length": len(timeline_labels),
        "score_count": len(scores), "simplified_truth": simplified,
        "ground_truth_intervals": intervals,
        # Typed arrays avoid duplicating every value in per-window histories.
        "final_indices_base64": packed(indices, "<f8"),
        "final_scores_base64": packed(scores, "<f8"),
        "final_labels_base64": packed(active, "u1"),
        "checking_windows": [[int(a), int(b), int(c), int(d)] for a, b, c, d in
                             zip(input_starts, input_ends, window_starts, window_ends)],
        "updates": [{"indices": [], "scores": []}],
    }
    save_decision_trace_html(trace, path)


def write_final_artifacts(directory, evaluation, test, labels, non_overlap):
    directory = Path(directory)
    labels = np.asarray(labels)
    covered = test["cover_counts"] > 0
    positions = np.flatnonzero(covered)

    def window_arrays(scored):
        targets = np.asarray([np.any(labels[start:end])
                              for start, end in zip(scored["starts"], scored["ends"])], dtype=int)
        return (scored["window_scores"], targets, scored["starts"], scored["ends"],
                {"input_start": scored["input_starts"], "input_end": scored["input_ends"]})

    arrays = {
        "window": window_arrays(test),
        "timeseries": (test["timeseries_scores"][covered], labels[covered],
                       positions, positions + 1, {"cover_count": test["cover_counts"][covered]}),
        "non_overlap_window": window_arrays(non_overlap),
    }
    artifacts = {}
    for protocol, section in MODES.items():
        metrics = evaluation[protocol]
        row = evaluation_row(metrics)
        with (directory / f"{section}_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        scores, targets, starts, ends, extras = arrays[protocol]
        threshold = metrics["calibrated"]["threshold"]
        with (directory / f"{section}_scores.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["start", "end", "score", "label", *extras,
                             "calibrated_prediction", "oracle_prediction"])
            for index, (start, end, score, target) in enumerate(zip(starts, ends, scores, targets)):
                writer.writerow([int(start), int(end), float(score), int(target),
                                 *(int(values[index]) for values in extras.values()),
                                 int(score > threshold), int(score > metrics["oracle"]["threshold"])])
        path = directory / f"{section}.html"
        source = non_overlap if protocol == "non_overlap_window" else test
        write_score_html(path, scores, targets, threshold, section,
                         timestep_indices=starts, series_labels=labels,
                         window_starts=source["starts"], window_ends=source["ends"],
                         input_starts=source["input_starts"], input_ends=source["input_ends"])
        artifacts[protocol] = str(path)
    return artifacts
