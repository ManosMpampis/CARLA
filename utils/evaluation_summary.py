"""Readable per-job and equal-machine summaries for inference-only sweeps."""

import csv
from pathlib import Path


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
