"""Plain-Python assertions for split-aware saved-result aggregation."""

import csv
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aggregate_tsb_ad_summaries import METRICS, MODES, SPLITS, summarize


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


with TemporaryDirectory() as temporary:
    root = Path(temporary)
    dataset = root / "dataset"
    dataset.mkdir()
    names = [f"{i}_Sensor_tr_{steps}_1st_900.csv" for i, steps in enumerate([500, 501, 1000, 2000])]
    for split, filename in SPLITS.items():
        selected = names if split == "all" else names[1:2] if split == "tuning" else [names[0], *names[2:]]
        (dataset / filename).write_text("file_name\n" + "\n".join(selected) + "\n")
    entries = []
    for experiment in ("working", "failed"):
        for index, name in enumerate(names):
            if experiment == "working" and index == 3:
                continue  # Missing runs stay in the expected denominator.
            metrics = root / f"{experiment}_{index}.json"
            values = {source: {metric: index / 10 for metric in ("f1", "event_f1", "range_f1", "affiliation_f1", "pa_f1")}
                      for source in ("oracle", "calibrated")}
            values.update(vus_pr=index / 10, vus_roc=index / 10, auc_pr=index / 10, auc_roc=index / 10)
            if index == 2:
                values["calibrated"]["affiliation_f1"] = None
                values["auc_pr"] = float("nan")
            metrics.write_text(json.dumps({"evaluation": {mode: values for mode in MODES.values()}}))
            entries.append(dict(experiment=experiment, file=name,
                                status="failed" if experiment == "failed" else "completed",
                                metrics_path=str(metrics), error="fixture failure" if experiment == "failed" else ""))
    source = root / "benchmark_runs" / "fixture" / "summary.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(entries))
    out = root / "out"
    manifest = summarize(root, dataset, "fixture", out)
    assert manifest["splits"]["all"] == dict(listed=4, eligible=3, excluded=1)
    for split, expected in (("all", 3), ("tuning", 1), ("eval", 2)):
        for mode in MODES:
            rows = {r["experiment"]: r for r in read(out / split / f"{mode}_summary.csv")}
            working, failed = rows["working"], rows["failed"]
            assert int(working["files_expected"]) == expected
            assert failed["oracle_f1_mean"] == "" and failed["oracle_f1_count"] == "0"
            assert int(failed["runs_failed"]) == expected
            assert all(f"{name}_mean" in working for name in METRICS)
            assert len(read(out / split / f"{mode}_files.csv")) == expected * 2
    result = {r["experiment"]: r for r in read(out / "all" / "timeseries_summary.csv")}["working"]
    assert result["runs_completed"] == "2" and result["runs_missing"] == "1"
    assert abs(float(result["oracle_f1_mean"]) - .15) < 1e-12
    assert abs(float(result["oracle_f1_std"]) - .07071067811865475) < 1e-12
    assert result["calibrated_affiliation_f1_count"] == "1"
    assert result["auc_pr_count"] == "1"
    entries.append(entries[0])
    source.write_text(json.dumps(entries))
    try:
        summarize(root, dataset, "fixture", out)
    except ValueError as exc:
        assert "Duplicate benchmark entry" in str(exc)
    else:
        raise AssertionError("Duplicate entries must be rejected")
print("TSB-AD summary assertions passed")
