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
    from utils.evaluation_logging import evaluation_row
    for procedure, values in row.get("evaluation", {}).items():
        result.update({f"{procedure}_{name}": value
                       for name, value in evaluation_row(values).items()})
    return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    fields = [name for name in fields if "_pa_f1" not in name] + [
        name for name in fields if "_pa_f1" in name]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_mode_csvs(root, rows, suffix="summary"):
    """One CSV per mode; shared provenance appears once per evaluation row."""
    from utils.evaluation_logging import MODES, evaluation_row
    for protocol, section in MODES.items():
        output = []
        for row in rows:
            info = {key: value for key, value in row.items()
                    if key not in {"evaluation", "evaluation_settings", "inference_speed"}}
            # Legacy headline fields belong only to their own evaluation mode.
            for name in ("timeseries_f1", "window_f1", "vus_pr", "vus_roc"):
                info.pop(name, None)
            info.update(row.get("evaluation_settings", {}))
            info.update({f"inference_{k}": v for k, v in row.get("inference_speed", {}).items()})
            values = row.get("evaluation", {}).get(protocol)
            output.append(evaluation_row(values, **info) if values else info)
        write_csv(Path(root) / f"{section}_{suffix}.csv", output)


def write_evaluation_overview(root):
    """Equal-series aggregates, kept separate for every evaluation mode."""
    from utils.evaluation_logging import MODES, evaluation_row
    root = Path(root)
    rows = [row for path in sorted(root.glob("*/summary.json"))
            for row in json.loads(path.read_text())]
    if not rows:
        return
    (root / "summary.json").write_text(json.dumps(rows, indent=2))
    write_mode_csvs(root, rows)
    groups = {}
    for row in rows:
        key = (row["experiment_key"], row["checkpoint_name"], row["case"])
        groups.setdefault(key, []).append(row)
    for protocol, section in MODES.items():
        aggregated = []
        for (experiment, checkpoint, case), members in sorted(groups.items()):
            complete = [r for r in members if r["status"] == "completed"]
            result = dict(experiment_key=experiment, checkpoint_name=checkpoint, case=case,
                          series_present=len(members), series_completed=len(complete))
            values = [{**evaluation_row(r["evaluation"][protocol]),
                       **{f"inference_{k}": v for k,v in r.get("inference_speed", {}).items()
                          if isinstance(v, (int,float))}} for r in complete
                      if protocol in r.get("evaluation", {})]
            names = list(dict.fromkeys(k for value in values for k in value
                                      if k not in {"calibrated_threshold", "oracle_threshold"}))
            for name in names:
                samples = [float(v[name]) for v in values
                           if v.get(name) is not None and math.isfinite(float(v[name]))]
                result[f"{name}_count"] = len(samples)
                result[f"{name}_mean"] = statistics.mean(samples) if samples else ""
                result[f"{name}_std"] = statistics.stdev(samples) if len(samples)>1 else 0.0 if samples else ""
            aggregated.append(result)
        write_csv(root / f"{section}_summary_aggregated.csv", aggregated)
        for case in sorted({row["case"] for row in rows}):
            write_csv(root / "cases" / case / f"{section}_summary_aggregated.csv",
                      [r for r in aggregated if r["case"] == case])
    for case in sorted({row["case"] for row in rows}):
        write_mode_csvs(root / "cases" / case, [r for r in rows if r["case"] == case])
