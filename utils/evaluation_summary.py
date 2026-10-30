"""Readable per-job and equal-machine summaries for inference-only sweeps."""

import csv
import json
import math
from pathlib import Path
import statistics


def flatten_evaluation(row):
    result = {key: value for key, value in row.items()
              if key not in {"evaluation", "evaluation_settings", "inference_speed"}}
    result.update(row.get("evaluation_settings", {}))
    for key, value in row.get("inference_speed", {}).items():
        result[f"inference_{key}"] = value
    for procedure, values in row.get("evaluation", {}).items():
        for source in ("calibrated", "oracle"):
            for name in ("f1_no_pa", "event_f1_no_pa", "threshold", "calibration_quantile"):
                if name in values.get(source, {}):
                    result[f"{procedure}_{source}_{name}"] = values[source][name]
        for name in ("vus_pr", "vus_roc"):
            if name in values:
                result[f"{procedure}_{name}"] = values[name]
        # Retain the original summary column names for existing readers.
        if "calibrated" in values:
            result[f"{procedure}_f1"] = values["calibrated"]["f1_no_pa"]
    for name in ("vus_pr", "vus_roc"):
        if name in row.get("evaluation", {}).get("timeseries", {}):
            result[name] = row["evaluation"]["timeseries"][name]
    return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_overview(root):
    """Merge machine reports without mixing different input cases or checkpoints."""
    root = Path(root)
    rows = [row for path in sorted(root.glob("machine-*/summary.json"))
            for row in json.loads(path.read_text())]
    if not rows:
        return
    flat = [flatten_evaluation(row) for row in rows]
    write_csv(root / "summary.csv", flat)
    (root / "summary.json").write_text(json.dumps(rows, indent=2))
    cases = sorted({row["case"] for row in flat})
    groups = {}
    for row in flat:
        key = (row["experiment_key"], row["checkpoint_name"], row["case"])
        groups.setdefault(key, []).append(row)
    aggregated = []
    for (experiment, checkpoint, case), members in sorted(groups.items()):
        complete = [row for row in members if row["status"] == "completed"]
        result = dict(experiment_key=experiment, checkpoint_name=checkpoint, case=case,
                      machines_present=len(members), machines_completed=len(complete))
        for name in ("input_window", "context_length", "target_length", "stride"):
            if complete and name in complete[0]:
                result[name] = complete[0][name]
        metrics = {name for row in complete for name in row
                   if name.endswith(("_f1", "_f1_no_pa", "_vus_pr", "_vus_roc"))}
        metrics.update({"inference_windows_per_second", "inference_ms_per_window",
                        "inference_batch_ms", "inference_input_timesteps_per_second"})
        for name in sorted(metrics):
            values = [float(row[name]) for row in complete
                      if name in row and math.isfinite(float(row[name]))]
            result[f"{name}_count"] = len(values)
            result[f"{name}_mean"] = statistics.mean(values) if values else ""
            result[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else ""
        aggregated.append(result)
    write_csv(root / "summary_aggregated.csv", aggregated)
    for case in cases:
        write_csv(root / "cases" / case / "summary.csv", [r for r in flat if r["case"] == case])
        write_csv(root / "cases" / case / "summary_aggregated.csv",
                  [r for r in aggregated if r["case"] == case])
