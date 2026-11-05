#!/usr/bin/env python3
"""Summarize saved TSB-AD metrics with equal weight per eligible series.

Reads final metrics.json via benchmark_runs/<version>/summary.json; does not
select epochs, retune thresholds, or recompute metrics. Outputs nine aggregate
CSVs (three splits x three modes), matching per-file CSVs, and coverage audits.
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import statistics

from utils.evaluation_logging import DECISION_METRICS, SCORE_METRICS

MODES = {"non_overlap": "non_overlap_window", "window": "window", "timeseries": "timeseries"}
METRICS = tuple(f"{source}_{name}" for source in ("oracle", "calibrated")
                for name in ("f1", "event_f1", "range_f1", "affiliation_f1", "pa_f1")) + SCORE_METRICS
SPLITS = {"all": "TSB-AD-M.csv", "tuning": "TSB-AD-M-Tuning.csv", "eval": "TSB-AD-M-Eva.csv"}


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_splits(dataset, min_train_steps):
    splits = {}
    for split, filename in SPLITS.items():
        with (dataset / filename).open(newline="", encoding="utf-8-sig") as stream:
            names = [row["file_name"] for row in csv.DictReader(stream)]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate filenames in {filename}")
        splits[split] = set(names)
    if splits["tuning"] & splits["eval"] or splits["tuning"] | splits["eval"] != splits["all"]:
        raise ValueError("Tuning and eval must partition the all-files CSV")
    lengths = {}
    for name in splits["all"]:
        match = re.search(r"_tr_(\d+)_1st_\d+\.csv$", name)
        if not match:
            raise ValueError(f"Cannot infer training length: {name}")
        lengths[name] = int(match[1])
    eligible = {split: {name for name in names if lengths[name] > min_train_steps}
                for split, names in splits.items()}
    return splits, eligible, lengths


def numeric(value):
    if value is None or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def summarize(base, dataset, version, output, min_train_steps=500):
    splits, eligible, lengths = read_splits(dataset, min_train_steps)
    summary_path = base / "benchmark_runs" / version / "summary.json"
    entries = json.loads(summary_path.read_text())
    indexed = {}
    for entry in entries:
        key = (entry["experiment"], entry["file"])
        if key in indexed:
            raise ValueError(f"Duplicate benchmark entry: {key}")
        indexed[key] = entry
    experiments = sorted({entry["experiment"] for entry in entries})
    if not experiments:
        raise ValueError(f"No experiments in {summary_path}")
    # Load each report only once, and only for a completed eligible job.
    reports = {}
    for key, entry in indexed.items():
        if entry["file"] not in eligible["all"] or entry["status"] != "completed":
            continue
        path = Path(entry.get("metrics_path") or Path(entry["directory"]) / "metrics.json")
        if not path.is_absolute() and not path.is_file():
            path = base / path
        try:
            reports[key] = (json.loads(path.read_text()), str(path))
        except (OSError, ValueError) as exc:
            reports[key] = ({}, str(path))
            reports[key][0]["summary_error"] = f"{type(exc).__name__}: {exc}"
    counts = {}
    detail_fields = ["experiment", "file", "train_steps", "status", "error", "metrics_path", *METRICS]
    aggregate_fields = ["experiment", "files_expected", "files_present", "runs_completed",
                        "runs_failed", "runs_missing", "runs_other", "files_with_mode"]
    aggregate_fields += [f"{name}_{stat}" for name in METRICS for stat in ("count", "mean", "std")]
    for split, filenames in eligible.items():
        counts[split] = {"listed": len(splits[split]), "eligible": len(filenames),
                         "excluded": len(splits[split]) - len(filenames)}
        for mode, protocol in MODES.items():
            details, aggregates = [], []
            for experiment in experiments:
                members = []
                statuses = Counter()
                for filename in sorted(filenames):
                    key = (experiment, filename)
                    entry = indexed.get(key, {})
                    status = entry.get("status", "missing")
                    statuses[status] += 1
                    report, path = reports.get(key, ({}, entry.get("metrics_path", "")))
                    values = report.get("evaluation", {}).get(protocol, {}) if status == "completed" else {}
                    error = entry.get("error", "") or report.get("summary_error", "")
                    if status == "completed" and not values and not error:
                        error = f"Missing evaluation mode: {protocol}"
                    detail = dict(experiment=experiment, file=filename, train_steps=lengths[filename],
                                  status=status, error=error, metrics_path=path)
                    for source in ("oracle", "calibrated"):
                        for name in DECISION_METRICS:
                            detail[f"{source}_{name}"] = numeric(values.get(source, {}).get(name))
                    for name in SCORE_METRICS:
                        detail[name] = numeric(values.get(name))
                    details.append(detail)
                    members.append((detail, bool(values)))
                result = dict(experiment=experiment, files_expected=len(filenames),
                              files_present=len(filenames) - statuses["missing"],
                              runs_completed=statuses["completed"], runs_failed=statuses["failed"],
                              runs_missing=statuses["missing"],
                              runs_other=sum(n for s, n in statuses.items() if s not in {"completed", "failed", "missing"}),
                              files_with_mode=sum(has_mode for _, has_mode in members))
                for name in METRICS:
                    samples = [row[name] for row, _ in members if row[name] is not None]
                    result.update({f"{name}_count": len(samples),
                                   f"{name}_mean": statistics.mean(samples) if samples else None,
                                   f"{name}_std": statistics.stdev(samples) if len(samples) > 1 else 0.0 if samples else None})
                aggregates.append(result)
            write_csv(output / split / f"{mode}_summary.csv", aggregates, aggregate_fields)
            write_csv(output / split / f"{mode}_files.csv", details, detail_fields)
    inventory = [dict(file=name, split="tuning" if name in splits["tuning"] else "eval",
                      train_steps=lengths[name], included=name in eligible["all"])
                 for name in sorted(splits["all"])]
    write_csv(output / "file_inventory.csv", inventory, ["file", "split", "train_steps", "included"])
    failures = [dict(experiment=exp, file=name, status=entry["status"], error=entry.get("error", ""))
                for (exp, name), entry in sorted(indexed.items())
                if name in eligible["all"] and entry["status"] != "completed"]
    write_csv(output / "failed_runs.csv", failures, ["experiment", "file", "status", "error"])
    manifest = dict(version=version, source_summary=str(summary_path.resolve()),
                    dataset=str(dataset.resolve()), min_train_steps_exclusive=min_train_steps,
                    aggregation="Equal-series arithmetic mean; sample std (ddof=1), singleton std=0. Missing/nonfinite metrics excluded independently; never filled with zero.",
                    metric_source="Final saved metrics.json evaluation; oracle and calibrated values retained as stored (no threshold or epoch selection).",
                    splits=counts, experiments=experiments, modes=MODES,
                    ignored_benchmark_files=sorted({r["file"] for r in entries} - splits["all"]))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("results/tsb_ad_m"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/TSB-AD-M"))
    parser.add_argument("--version", default="first_run")
    parser.add_argument("--min-train-steps", type=int, default=500, help="Exclusive lower bound")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("AGENTIC_DOCS/reports/tsb_ad_m_summaries") / args.version
    manifest = summarize(args.base, args.dataset, args.version, output, args.min_train_steps)
    print(json.dumps(manifest["splits"], indent=2))
    print(f"Wrote nine summaries for {len(manifest['experiments'])} experiments -> {output}")


if __name__ == "__main__":
    main()
