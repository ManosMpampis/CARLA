"""Evaluate exported benchmark scores without importing CARLA's model stack.

Run with Python 3.11/3.12 and TSB-AD dependencies when the training environment
cannot import the official benchmark package.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

from utils.tsb_ad_metrics import METRIC_NAMES, evaluate_scores, json_values


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_values(data), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _input_path(value, result_path: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Missing exported score or evaluation-input path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else result_path.parent / path


def _algorithm_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "experiment"
    if safe != name:
        safe += "_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return safe + ".csv"


def _result_paths(root: Path) -> list[Path]:
    """Resolve a version summary, or discover result files in an output root."""
    summary_path = root / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary, list):
            paths = []
            for row in summary:
                if not isinstance(row, dict) or not isinstance(row.get("directory"), str):
                    raise ValueError("Benchmark summary rows must contain directory paths: " + str(summary_path))
                directory = Path(row["directory"]).expanduser()
                if not directory.is_absolute():
                    directory = root / directory
                paths.append((directory / "benchmark_result.json").resolve())
            return sorted(set(paths))
    return sorted(root.rglob("benchmark_result.json"))


def _reject_duplicate_series(result_paths: list[Path]) -> None:
    """Avoid mixing repeated evaluations from different benchmark versions."""
    seen = {}
    for result_path in result_paths:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # The evaluator records malformed or missing exports.
        if not isinstance(result, dict) or result.get("status") != "completed":
            continue
        experiment, filename = result.get("experiment"), result.get("file")
        if not isinstance(experiment, str) or not isinstance(filename, str):
            continue
        key = (experiment, filename)
        if key in seen:
            raise ValueError(
                "Duplicate completed experiment/series " + repr(key)
                + " at " + str(seen[key]) + " and " + str(result_path)
                + ". Select one benchmark version with --run-dir "
                + "<output>/benchmark_runs/<version>."
            )
        seen[key] = result_path


def evaluate_run(
    run_dir: str | Path,
    *,
    tsb_ad_root: str | Path | None = None,
    sliding_window: int | None = None,
) -> dict:
    """Evaluate completed benchmark_result.json exports and write CSV summaries."""
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Run directory does not exist: " + str(root))
    result_paths = _result_paths(root)
    if not result_paths:
        raise ValueError("No benchmark_result.json exports found under " + str(root))
    _reject_duplicate_series(result_paths)
    rows_by_algorithm = {}
    evaluated = []
    failures = []
    skipped = []
    for result_path in result_paths:
        result = None
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ValueError("Benchmark result must be a JSON object")
            if result.get("status") != "completed":
                skipped.append({"result_path": str(result_path), "status": result.get("status")})
                continue
            experiment = result.get("experiment")
            filename = result.get("file")
            if not isinstance(experiment, str) or not experiment:
                raise ValueError("Completed result is missing its experiment name")
            if not isinstance(filename, str) or not filename:
                raise ValueError("Completed result is missing its series filename")
            rows_by_algorithm.setdefault(experiment, [])
            score_path = _input_path(result.get("score_path"), result_path)
            inputs_path = _input_path(result.get("evaluation_inputs_path"), result_path)
            scores = np.load(score_path, allow_pickle=False)
            if not isinstance(scores, np.ndarray):
                scores.close()
                raise ValueError("score_path must point to a .npy array")
            with np.load(inputs_path, allow_pickle=False) as inputs:
                metrics = evaluate_scores(
                    scores,
                    inputs["labels"],
                    inputs["first_channel"],
                    tsb_ad_root=tsb_ad_root,
                    sliding_window=sliding_window,
                )
            metrics_path = result_path.with_name("tsb_metrics.json")
            _write_json(metrics_path, metrics)
            result["metric_status"] = "completed"
            result["tsb_metrics_path"] = str(metrics_path)
            result.pop("metric_error", None)
            _write_json(result_path, result)
            detector_seconds = result.get("detector_seconds")
            if detector_seconds is None:
                detector_seconds = result.get("elapsed_seconds")
            row = {"file": filename, "Time": detector_seconds}
            row.update({name: metrics.get(name) for name in METRIC_NAMES})
            rows_by_algorithm[experiment].append(row)
            evaluated.append({
                "experiment": experiment,
                "file": filename,
                "result_path": str(result_path),
                "metrics_path": str(metrics_path),
            })
            print("Evaluated " + experiment + ": " + filename)
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
            failure = {"result_path": str(result_path), "error": error}
            if isinstance(result, dict):
                failure.update({"experiment": result.get("experiment"), "file": result.get("file")})
                result["metric_status"] = "error"
                result["metric_error"] = error
                result.pop("tsb_metrics_path", None)
                try:
                    _write_json(result_path, result)
                except OSError as write_exc:
                    failure["write_error"] = str(write_exc)
            failures.append(failure)
            print("Failed " + str(result_path) + ": " + error, file=sys.stderr)
    summary_dir = root / "benchmark_metrics"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = {}
    for algorithm, rows in sorted(rows_by_algorithm.items()):
        csv_path = summary_dir / _algorithm_filename(algorithm)
        temporary = csv_path.with_name(csv_path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("file", "Time", *METRIC_NAMES))
            writer.writeheader()
            writer.writerows(json_values(rows))
        temporary.replace(csv_path)
        csv_paths[algorithm] = str(csv_path)
    summary = {
        "run_dir": str(root),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "tsb_ad_root": str(Path(tsb_ad_root).expanduser().resolve()) if tsb_ad_root is not None else None,
        "sliding_window_override": sliding_window,
        "threshold_protocol": "official_oracle",
        "evaluated_count": len(evaluated),
        "failure_count": len(failures),
        "skipped_count": len(skipped),
        "csv_paths": csv_paths,
        "evaluated": evaluated,
        "failures": failures,
        "skipped": skipped,
    }
    _write_json(summary_dir / "summary.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Named benchmark_runs/version directory, or output root with unique series results")
    parser.add_argument("--tsb-ad-root", help="Optional TSB-AD source checkout; otherwise use installed package")
    parser.add_argument("--sliding-window", type=int, help="Override official first-channel metric period")
    args = parser.parse_args(argv)
    if args.sliding_window is not None and args.sliding_window < 1:
        parser.error("--sliding-window must be positive")
    try:
        summary = evaluate_run(
            args.run_dir,
            tsb_ad_root=args.tsb_ad_root,
            sliding_window=args.sliding_window,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "Official TSB-AD metrics: " + str(summary["evaluated_count"]) + " evaluated, "
        + str(summary["failure_count"]) + " failed, "
        + str(summary["skipped_count"]) + " skipped"
    )
    return 1 if summary["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
