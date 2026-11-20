"""Run ./venv/bin/python tests/check_evaluator_artifacts.py (uses temporary output)."""

import base64
from copy import deepcopy
import csv
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluator import Evaluator


class Model(torch.nn.Module):
    def __init__(self, crop=0):
        super().__init__()
        self.crop = crop
        self.calls = 0

    def score(self, x):
        self.calls += 1
        return {"fused": x[:, 0, self.crop:]}


def read_report(path):
    html = Path(path).read_text(encoding="utf-8")
    encoded = re.search(r'<script id="evaluation-data" type="text/plain">([^<]+)</script>', html).group(1)
    assert "__EVALUATION_DATA__" not in html
    assert '<script src=' not in html
    return json.loads(base64.b64decode(encoded))


def array(report, name, dtype):
    return np.frombuffer(base64.b64decode(report["arrays"][name]), dtype=dtype)


def verify(evaluator, results):
    for name, files in evaluator.artifacts.items():
        target = evaluator.aggregated[name]["test"]
        keep = target["cover_counts"] > 0
        scores, labels = target["scores"][keep], target["labels"][keep]
        intervals, indices = target["intervals"][keep], target["indices"][keep]
        with Path(files["scores_csv"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(scores)
        np.testing.assert_array_equal([int(row["index"]) for row in rows], indices)
        np.testing.assert_array_equal([[int(row["start"]), int(row["end"])] for row in rows], intervals)
        np.testing.assert_array_equal([int(row["label"]) for row in rows], labels)
        np.testing.assert_array_equal([int(row["cover_count"]) for row in rows], target["cover_counts"][keep])
        np.testing.assert_array_equal([float(row["score"]) for row in rows], scores)
        assert all(row["axis"] == target["axis"] for row in rows)
        report = read_report(files["html"])
        assert report["name"] == name and report["axis"] == target["axis"]
        assert report["prediction_operator"] == ">="
        np.testing.assert_array_equal(array(report, "scores", "<f4"), scores)
        np.testing.assert_array_equal(array(report, "indices", "<f8"), indices)
        np.testing.assert_array_equal(array(report, "intervals", "<f8").reshape(-1, 2), intervals)
        np.testing.assert_array_equal(array(report, "labels", "u1"), labels)
        for source in ("calibrated", "oracle"):
            expected = (scores >= results[name][source]["threshold"]).astype(int)
            np.testing.assert_array_equal([int(row[source + "_prediction"]) for row in rows], expected)
            np.testing.assert_array_equal(array(report, source, "u1"), expected)
        with Path(files["metrics_csv"]).open(newline="") as stream:
            metrics = list(csv.DictReader(stream))
        assert [row["threshold_source"] for row in metrics] == ["calibrated", "oracle"]
        for row in metrics:
            source = row.pop("threshold_source")
            for key, value in results[name][source].items():
                if value is None:
                    assert row[key] == ""
                    assert report["metrics"][source][key] is None
                elif np.isfinite(value):
                    assert float(row[key]) == value


train = (np.arange(17, dtype=np.float32) / 100)[:, None]
test = (np.arange(17, dtype=np.float32) / 10)[:, None]
labels = np.zeros(17, dtype=np.int32)
labels[7:10] = 1
presets = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/evaluator.yml").read_text())
base = {"wsz": 6, "stride": 4, "eval_window_size": 2,
        "evaluator": {"metrics": ["f1_score", "pa_f1_score", "VUS_ROC"]}}

with TemporaryDirectory(prefix="evaluator-artifacts-") as temporary:
    root = Path(temporary)
    evaluator, model = Evaluator(base), Model()
    results = evaluator(model, train, test, test_labels=labels)
    assert evaluator.artifacts == {} and not list(root.iterdir())
    calls = model.calls
    paths = evaluator.write_artifacts(root / "manual")
    assert model.calls == calls, "export must not run inference"
    assert set(paths) == {"window", "timeseries", "non_overlap_window"}
    verify(evaluator, results)

    results = evaluator(model, train, test, test_labels=labels, epoch=7, output_dir=root / "automatic")
    assert Path(evaluator.artifacts["window"]["html"]) == root / "automatic/epoch_000007/window/report.html"
    verify(evaluator, results)
    previous = Path(evaluator.artifacts["window"]["html"]).read_bytes()
    evaluator(model, train, test, test_labels=labels, epoch=8, output_dir=root / "automatic")
    assert (root / "automatic/epoch_000007/window/report.html").read_bytes() == previous
    assert (root / "automatic/epoch_000008/window/report.html").exists()

    # Arbitrary modes, cropped leading outputs with duplicate original starts,
    # and automatic tag_dir fallback all use the same export path.
    config = deepcopy(base)
    config["stride"] = 1
    config["tag_dir"] = str(root / "run")
    config["evaluation"] = {"check_mask": [2, 6], "latency_offset": -6}
    config["evaluator"]["aggregators"] = {"sum window α": {
        **presets["evaluator"]["aggregators"]["window"], "output_fn": "sum"}}
    config["evaluator"]["artifacts"] = {"enabled": True}
    evaluator = Evaluator(config)
    results = evaluator(Model(2), train, test, test_labels=labels, epoch=2)
    verify(evaluator, results)
    target = evaluator.aggregated["sum window α"]["test"]
    assert np.unique(target["intervals"][:, 0]).size < len(target["intervals"])
    assert Path(evaluator.artifacts["sum window α"]["html"]) == root / "run/evaluation/epoch_000002/sum window α/report.html"

    # Interior gaps are omitted from both exported formats.
    config = {**base, "stride": 8}
    evaluator = Evaluator(config)
    results = evaluator(Model(), train, test, test_labels=labels, output_dir=root / "gaps")
    verify(evaluator, results)
    report = read_report(evaluator.artifacts["timeseries"]["html"])
    assert 6 not in array(report, "indices", "<f8")

    # Undefined single-class metrics serialize as blank CSV fields / JSON null.
    evaluator = Evaluator(base)
    results = evaluator(Model(), train, test, test_labels=np.zeros(17), output_dir=root / "normal")
    verify(evaluator, results)

    for csv_enabled, html_enabled in ((True, False), (False, True)):
        config = deepcopy(base)
        config["evaluator"]["artifacts"] = {"csv": csv_enabled, "html": html_enabled}
        evaluator = Evaluator(config)
        evaluator(Model(), train, test, test_labels=labels, output_dir=root / f"formats-{csv_enabled}")
        for files in evaluator.artifacts.values():
            assert ("html" in files) == html_enabled
            assert ("scores_csv" in files) == csv_enabled
            assert ("metrics_csv" in files) == csv_enabled
            directory = Path(next(iter(files.values()))).parent
            assert {path.name for path in directory.iterdir()} == (
                ({"scores.csv", "metrics.csv"} if csv_enabled else set()) | ({"report.html"} if html_enabled else set()))

    evaluator = Evaluator(base)
    try:
        evaluator.write_artifacts(root / "not-evaluated")
    except ValueError:
        pass
    else:
        raise AssertionError("export before evaluation must fail")
    assert not (root / "not-evaluated").exists()

    config = deepcopy(base)
    config["evaluator"]["aggregators"] = {"../outside": presets["evaluator"]["aggregators"]["window"]}
    model = Model()
    try:
        Evaluator(config)(model, train, test, test_labels=labels, output_dir=root)
    except ValueError as error:
        assert "single directory names" in str(error)
    else:
        raise AssertionError("artifact mode must not escape its output directory")
    assert model.calls == 0

print("CSV/HTML exports, thresholds, custom modes, epochs, gaps, format switches, and no inference on export passed.")
