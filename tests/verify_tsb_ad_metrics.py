"""Verify official-metric handoff and deferred evaluation without downloads.

Optional: --tsb-ad-root /path/to/TSB-AD also compares against real upstream.
"""

import argparse
import ast
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.tsb_ad_metrics import evaluate_scores, json_values, load_metrics


EXPECTED = {
    "AUC-PR": 0.11,
    "AUC-ROC": 0.22,
    "VUS-PR": 0.33,
    "VUS-ROC": 0.44,
    "Standard-F1": 0.55,
    "PA-F1": 0.66,
    "Event-based-F1": 0.77,
    "R-based-F1": 0.88,
    "Affiliation-F": None,
}
SCORES = np.array([-2.0, 0.0, 0.5, 2.5])
LABELS = np.array([0, 1, 0, 1], dtype=np.int64)
CHANNEL = np.array([3.0, 1.0, 4.0, 2.0])


def rejected(action, message):
    try:
        action()
    except ValueError as exc:
        assert message in str(exc), (message, str(exc))
    else:
        raise AssertionError("Expected ValueError containing " + repr(message))


def verify_api():
    calls = []

    def get_metrics(scores, labels, slidingWindow=100, pred=None):
        assert np.array_equal(scores, SCORES)
        assert np.array_equal(labels, LABELS)
        calls.append((slidingWindow, pred))
        return {name: np.float64(value if value is not None else np.nan)
                for name, value in EXPECTED.items()}

    def find_length_rank(data, rank=1):
        assert data.shape == (4, 1) and rank == 1
        assert np.array_equal(data[:, 0], CHANNEL)
        return np.int64(7)

    metric_module = SimpleNamespace(get_metrics=get_metrics)
    window_module = SimpleNamespace(find_length_rank=find_length_rank)

    def import_module(name):
        if name == "TSB_AD.evaluation.metrics":
            return metric_module
        if name == "TSB_AD.utils.slidingWindows":
            return window_module
        raise AssertionError("Unexpected import: " + name)

    with patch("utils.tsb_ad_metrics.importlib.import_module", side_effect=import_module):
        assert load_metrics() == (metric_module, window_module)
        assert evaluate_scores(SCORES, LABELS, CHANNEL) == EXPECTED
        assert calls[-1] == (7, None), "Default benchmark metrics must use oracle thresholds"
        fixed_predictions = np.array([0, 0, 1, 1])
        assert evaluate_scores(SCORES, LABELS, CHANNEL, predictions=fixed_predictions) == EXPECTED
        assert np.array_equal(calls[-1][1], fixed_predictions)
        assert calls[-1][0] == 7

    # An explicit metric buffer must not require the period-estimation module
    # or its optional dependencies.
    with patch("utils.tsb_ad_metrics.importlib.import_module", return_value=metric_module) as importer:
        assert evaluate_scores(SCORES, LABELS, CHANNEL, sliding_window=3) == EXPECTED
        assert importer.call_args_list == [(('TSB_AD.evaluation.metrics',), {})]
        assert calls[-1] == (3, None)

    # Reject corrupt or shifted timelines before attempting any upstream import.
    with patch("utils.tsb_ad_metrics.importlib.import_module", side_effect=AssertionError("Imported invalid inputs")):
        for bad_scores, message in (
            ([], "nonempty"),
            (np.ones((4, 2)), "one-dimensional"),
            (np.ones((1, 4, 1)), "one-dimensional"),
            (SCORES[:-1], "align"),
            ([1, 2, np.nan, 4], "finite"),
            ([1, 2, np.inf, 4], "finite"),
            ([1j, 2j, 3j, 4j], "real"),
            (["a", "b", "c", "d"], "numeric"),
        ):
            rejected(lambda: evaluate_scores(bad_scores, LABELS, CHANNEL), message)
        for bad_labels, message in (([0, 2, 0, 1], "0 or 1"), (LABELS[:-1], "align"), ([0, 1, np.nan, 1], "finite")):
            rejected(lambda: evaluate_scores(SCORES, bad_labels, CHANNEL), message)
        for bad_channel, message in ((CHANNEL[:-1], "align"), (np.ones((4, 2)), "one-dimensional"), ([0, 1, np.inf, 3], "finite")):
            rejected(lambda: evaluate_scores(SCORES, LABELS, bad_channel), message)
        for bad_window in (0, -1, True, np.bool_(True), 1.5, "3"):
            rejected(lambda: evaluate_scores(SCORES, LABELS, CHANNEL, sliding_window=bad_window), "positive integer")
        for bad_predictions, message in (([0, 2, 1, 0], "0 or 1"), ([0, 1], "align"), ([0, np.nan, 1, 0], "finite")):
            rejected(lambda: evaluate_scores(SCORES, LABELS, CHANNEL, predictions=bad_predictions), message)

    assert json_values({"array": np.array([np.inf, 0.5]), "int": np.int64(3),
                        "bool": np.bool_(True), "long": np.longdouble(np.nan)}) == {
        "array": [None, 0.5], "int": 3, "bool": True, "long": None,
    }
    assert "torch" not in sys.modules and "utils.config" not in sys.modules


def fake_upstream(root):
    for subdirectory in ("", "evaluation", "utils"):
        directory = root / "TSB_AD" / subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
    metric_values = dict(EXPECTED)
    metric_values["Affiliation-F"] = 0.0
    source = (
        "import numpy as np\n"
        "def get_metrics(score, labels, slidingWindow=100, pred=None):\n"
        "    np.testing.assert_array_equal(score, [-2.0, 0.0, 0.5, 2.5])\n"
        "    np.testing.assert_array_equal(labels, [0, 1, 0, 1])\n"
        "    assert pred is None, 'Saved predictions must not replace oracle metrics'\n"
        "    assert slidingWindow in (7, 3)\n"
        "    result = {key: np.float64(value) for key, value in " + repr(metric_values) + ".items()}\n"
        "    result['Affiliation-F'] = np.float64(np.nan)\n"
        "    return result\n"
    )
    (root / "TSB_AD/evaluation/metrics.py").write_text(source, encoding="utf-8")
    (root / "TSB_AD/utils/slidingWindows.py").write_text(
        "import numpy as np\n"
        "def find_length_rank(data, rank=1):\n"
        "    assert data.shape == (4, 1) and rank == 1\n"
        "    np.testing.assert_array_equal(data[:, 0], [3.0, 1.0, 4.0, 2.0])\n"
        "    return 7\n",
        encoding="utf-8",
    )


def export_case(root, name, *, experiment="test_arm", status="completed", scores=SCORES):
    directory = root / name
    directory.mkdir(parents=True)
    np.save(directory / "scores.npy", scores)
    np.savez(directory / "inputs.npz", labels=LABELS, first_channel=CHANNEL,
             predictions=np.ones(4, dtype=int))
    result = dict(status=status, experiment=experiment, file=name + ".csv",
                  score_path="scores.npy", evaluation_inputs_path="inputs.npz",
                  elapsed_seconds=1.25, metric_status="error", metric_error="old error")
    path = directory / "benchmark_result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return directory


def invoke(run_dir, upstream, *extra):
    return subprocess.run(
        [sys.executable, str(REPO / "evaluate_tsb_ad.py"), "--run-dir", str(run_dir),
         "--tsb-ad-root", str(upstream), *extra],
        cwd=REPO, capture_output=True, text=True,
    )


def verify_cli():
    with tempfile.TemporaryDirectory(prefix="tsb-ad-metrics-") as temporary:
        root = Path(temporary)
        upstream = root / "upstream"
        fake_upstream(upstream)
        run = root / "run"
        good = export_case(run, "good")
        skipped = export_case(run, "skipped", status="failed")
        process = invoke(run, upstream)
        assert process.returncode == 0, process.stderr
        assert json.loads((good / "tsb_metrics.json").read_text()) == EXPECTED
        result = json.loads((good / "benchmark_result.json").read_text())
        assert result["status"] == "completed" and result["metric_status"] == "completed"
        assert "metric_error" not in result
        assert Path(result["tsb_metrics_path"]) == good / "tsb_metrics.json"
        assert json.loads((skipped / "benchmark_result.json").read_text())["metric_error"] == "old error"
        csv_path = run / "benchmark_metrics/test_arm.csv"
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            assert reader.fieldnames == ["file", "Time", *EXPECTED]
        assert len(rows) == 1 and rows[0]["file"] == "good.csv"
        assert rows[0]["Time"] == "1.25" and rows[0]["VUS-PR"] == "0.33"
        assert rows[0]["Affiliation-F"] == "", "Undefined metrics must be empty CSV cells"

        missing = export_case(run, "missing", experiment="bad_arm")
        (missing / "scores.npy").unlink()
        shifted = export_case(run, "shifted", scores=SCORES[:-1])
        # Force the override path to succeed without the upstream window module.
        (upstream / "TSB_AD/utils/slidingWindows.py").unlink()
        process = invoke(run, upstream, "--sliding-window", "3")
        assert process.returncode == 1, process.stdout + process.stderr
        summary = json.loads((run / "benchmark_metrics/summary.json").read_text())
        assert summary["evaluated_count"] == 1 and summary["failure_count"] == 2
        assert summary["skipped_count"] == 1 and summary["threshold_protocol"] == "official_oracle"
        assert len(summary["failures"]) == 2
        for case in (missing, shifted):
            result = json.loads((case / "benchmark_result.json").read_text())
            assert result["status"] == "completed" and result["metric_status"] == "error"
            assert result["metric_error"] and "tsb_metrics_path" not in result
        with csv_path.open(newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 1
        with (run / "benchmark_metrics/bad_arm.csv").open(newline="") as stream:
            assert list(csv.DictReader(stream)) == [], "Failed runs must not become metric rows"
        empty = root / "empty"
        empty.mkdir()
        assert invoke(empty, upstream).returncode == 2

        # Models live outside benchmark_runs/<version>. Its summary is the
        # authoritative selection, so evaluating one version must not discover
        # or mutate the other version's repeated experiment/series pair.
        output = root / "versioned"
        first = export_case(output / "models/v1", "same", experiment="version_arm")
        second = export_case(output / "models/v2", "same", experiment="version_arm")
        first_result_path = first / "benchmark_result.json"
        first_result = json.loads(first_result_path.read_text())
        first_result["detector_seconds"] = 0.25
        first_result_path.write_text(json.dumps(first_result))
        version = output / "benchmark_runs/v1"
        version.mkdir(parents=True)
        version_summary = [{"directory": str(first), "status": "completed"}]
        (version / "summary.json").write_text(json.dumps(version_summary))
        process = invoke(output, upstream, "--sliding-window", "3")
        assert process.returncode == 2 and "Duplicate completed" in process.stderr
        assert "benchmark_runs/<version>" in process.stderr
        assert not (first / "tsb_metrics.json").exists(), "Duplicate rejection must precede mutation"
        process = invoke(version, upstream, "--sliding-window", "3")
        assert process.returncode == 0, process.stderr
        summary = json.loads((version / "benchmark_metrics/summary.json").read_text())
        assert summary["evaluated_count"] == 1 and summary["failure_count"] == 0
        assert json.loads((version / "summary.json").read_text()) == version_summary
        assert not (second / "tsb_metrics.json").exists()
        assert json.loads((second / "benchmark_result.json").read_text())["metric_status"] == "error"
        with (version / "benchmark_metrics/version_arm.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows[0]["Time"] == "0.25", "Benchmark Time must exclude metrics and trace overhead"

        # Relative directories in summaries resolve against the selected run.
        version_two = output / "benchmark_runs/v2"
        version_two.mkdir(parents=True)
        (version_two / "summary.json").write_text(json.dumps([
            {"directory": "../../models/v2/same", "status": "completed"}
        ]))
        process = invoke(version_two, upstream, "--sliding-window", "3")
        assert process.returncode == 0, process.stderr


def verify_live(tsb_ad_root):
    metric_module, window_module = load_metrics(tsb_ad_root)
    positions = np.arange(160)
    labels = np.zeros(160, dtype=np.int64)
    labels[100:115] = 1
    scores = np.abs(np.sin(positions / 13)) + labels * 2
    channel = np.sin(positions / 4)
    window = window_module.find_length_rank(channel.reshape(-1, 1), rank=1)
    expected = json_values(metric_module.get_metrics(scores, labels, slidingWindow=window, pred=None))
    assert evaluate_scores(scores, labels, channel, tsb_ad_root=tsb_ad_root) == expected
    print("verify_tsb_ad_metrics: real upstream equality OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsb-ad-root", help="Optional real upstream checkout for an additional comparison")
    args = parser.parse_args()
    for filename in ("evaluate_tsb_ad.py", "utils/tsb_ad_metrics.py"):
        ast.parse((REPO / filename).read_text(), filename=filename, feature_version=(3, 11))
    verify_api()
    verify_cli()
    if args.tsb_ad_root:
        verify_live(args.tsb_ad_root)
    print("verify_tsb_ad_metrics: OK (official handoff, validation, deferred CLI and failures)")


if __name__ == "__main__":
    main()
