"""Synthetic end-to-end checks: ./venv/bin/python tests/check_evaluator.py."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluator import Evaluator, evaluation_options
from utils.timeseries_aggregation import TimeseriesAggregator


class Model(torch.nn.Module):
    def __init__(self, crop=0):
        super().__init__()
        self.crop = crop
        self.seen = []

    def score(self, x):
        assert not torch.is_grad_enabled()
        self.seen.extend(x[:, 0, 0].cpu().tolist())
        return {"fused": x[:, 0, self.crop:]}


class Logger:
    def __init__(self):
        self.rows = []
        self.messages = []

    def metrics_summary(self, phase, values, step):
        self.rows.append((phase, values, step))

    def log(self, message):
        self.messages.append(message)


train = (np.arange(17, dtype=np.float32) / 100)[:, None]
test = (np.arange(17, dtype=np.float32) / 10)[:, None]
labels = np.zeros(17, dtype=np.int32)
labels[7:10] = 1
presets = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/evaluator.yml").read_text())


def starts(n, window, stride, include_last):
    result = set(range(0, n - window + 1, stride))
    if include_last:
        result.add(n - window)
    return result


def reference_scores(series, window, stride, crop, latency, include_last):
    """Independent placement oracle, with no legacy scoring module dependency."""
    contributions = [[] for _ in series]
    windows, bounds = [], []
    for start in sorted(starts(len(series), window, stride, include_last)):
        raw = series[start + crop:start + window, 0]
        positions = np.arange(start + crop + latency, start + window + latency)
        valid = (positions >= 0) & (positions < len(series))
        if not valid.any():
            continue
        windows.append(raw[valid].mean())
        bounds.append((positions[valid][0], positions[valid][-1] + 1))
        for position, value in zip(positions[valid], raw[valid]):
            contributions[position].append(value)
    return {"window_scores": np.array(windows), "starts": np.array([s for s, _ in bounds]),
            "ends": np.array([e for _, e in bounds]),
            "timeseries_scores": np.array([np.mean(values) if values else 0 for values in contributions]),
            "cover_counts": np.array([len(values) for values in contributions])}


for crop, latency, stride in ((0, 0, 4), (2, 1, 3), (2, -4, 3), (0, 0, 8)):
    config = {"wsz": 6, "stride": stride, "score_batch_size": 3,
              "eval_window_size": 2, "calibration_quantile": 0.75,
              "evaluation": {"check_mask": [crop, 6] if crop else None,
                             "latency_offset": latency}, **deepcopy(presets)}
    saved_config = deepcopy(config)
    model, logger = Model(crop), Logger()
    evaluator = Evaluator(config, device="cpu", logger=logger)
    initial_aggregators = {mode: splits.copy() for mode, splits in evaluator.aggregators.items()}
    assert set(initial_aggregators) == {"window", "timeseries", "non_overlap_window"}
    assert initial_aggregators["timeseries"]["train"].output_window is None
    # Calibration prefers the complete normal series to the shortened data split.
    train_dataset = SimpleNamespace(data=train[:8], full_train_series=train)
    test_dataset = SimpleNamespace(data=test, targets=labels)
    result = evaluator(model, train_dataset, test_dataset, epoch=7)
    assert config == saved_config
    assert model.training
    options = evaluation_options(config)
    expected_starts = starts(17, 6, stride, options["include_last_window"]) | starts(17, 6, 6-crop, False)
    expected_seen = [float(data[s, 0]) for data in (train, test) for s in sorted(expected_starts)]
    assert Counter(model.seen) == Counter(expected_seen), "each input must be scored once per split"
    assert len(logger.rows) == len(logger.messages) == 6
    assert all(row[2] == 7 for row in logger.rows)

    for mode in ("window", "timeseries", "non_overlap_window"):
        mode_stride = 6 - crop if mode == "non_overlap_window" else stride
        include_last = False if mode == "non_overlap_window" else options["include_last_window"]
        for split, data in (("train", train), ("test", test)):
            old = reference_scores(data, 6, mode_stride, crop, latency, include_last)
            current = evaluator.aggregated[mode][split]
            keep = current["cover_counts"] > 0
            if mode == "timeseries":
                covered = old["cover_counts"] > 0
                expected_scores = old["timeseries_scores"][covered]
                expected_labels = labels[covered]
                np.testing.assert_array_equal(current["indices"][keep], np.flatnonzero(covered))
                np.testing.assert_array_equal(current["cover_counts"][keep], old["cover_counts"][covered])
            else:
                expected_scores = old["window_scores"]
                expected_labels = np.array([labels[s:e].any() for s, e in zip(old["starts"], old["ends"])])
                np.testing.assert_array_equal(current["intervals"][keep], np.column_stack((old["starts"], old["ends"])))
            np.testing.assert_allclose(current["scores"][keep], expected_scores, rtol=1e-6, atol=1e-7)
            if split == "test":
                np.testing.assert_array_equal(current["labels"][keep], expected_labels)
        clean = evaluator.aggregated[mode]["train"]
        target = evaluator.aggregated[mode]["test"]
        test_scores = target["scores"][target["cover_counts"] > 0]
        test_labels = target["labels"][target["cover_counts"] > 0]
        assert result[mode]["calibrated"]["threshold"] == float(np.quantile(
            clean["scores"][clean["cover_counts"] > 0], 0.75))
        best_f1 = max(f1_score(test_labels, test_scores >= threshold, zero_division=0)
                      for threshold in np.unique(test_scores))
        assert np.isclose(result[mode]["oracle"]["f1_score"], best_f1)
        assert set(result[mode]["oracle"]) == set(presets["evaluator"]["metrics"]) | {"threshold"}
        assert evaluator.aggregators[mode]["train"] is not evaluator.aggregators[mode]["test"]
    assert evaluator.aggregators["timeseries"]["test"].output_window == 6 - crop
    # Reusing an evaluator resets accumulators instead of accumulating duplicates.
    second = evaluator(model, train, test, test_labels=labels)
    assert second == result
    for mode, splits in initial_aggregators.items():
        for split, aggregator in splits.items():
            assert evaluator.aggregators[mode][split] is aggregator
    print(f"Legacy placement, shared scoring, thresholds, logging, reset passed: crop={crop}, latency={latency}, stride={stride}")

# Arbitrary named aggregations select the number of modes; sum is configurable.
custom = {"wsz": 6, "stride": 4, "eval_window_size": 2, "evaluator": {
    "metrics": ["f1_score", "VUS_ROC"],
    "aggregators": {"sum_window": {**presets["evaluator"]["aggregators"]["window"], "output_fn": "sum"}}}}
evaluator = Evaluator(custom)
model = Model()
result = evaluator(model, train, test, test_labels=labels)
assert list(result) == ["sum_window"]
np.testing.assert_allclose(evaluator.aggregated["sum_window"]["test"]["scores"],
                           [test[s:s+6, 0].sum() for s in (0, 4, 8, 11)])
assert set(result["sum_window"]["oracle"]) == {"threshold", "f1_score", "VUS_ROC"}

# Reusing an inferred-width aggregator allows a different score map in a new run.
config = {"wsz": 6, "stride": 4, "eval_window_size": 2, "evaluator": {
    "aggregators": {"timeseries": {**presets["evaluator"]["aggregators"]["timeseries"],
                                  "time_offset": 0}}}}
dynamic_evaluator = Evaluator(config)
dynamic_aggregator = dynamic_evaluator.aggregators["timeseries"]["test"]
for crop in (0, 2):
    dynamic_evaluator(Model(crop), train, test, test_labels=labels)
    snapshot = dynamic_evaluator.aggregated["timeseries"]["test"]
    assert dynamic_evaluator.aggregators["timeseries"]["test"] is dynamic_aggregator
    assert dynamic_aggregator.output_window == 6 - crop
    np.testing.assert_array_equal(snapshot["indices"], np.arange(len(test) - crop))

# Precomputed-score aggregation still agrees with the direct model path.
aggregator = TimeseriesAggregator(6, 4, 6, include_last_window=True, gap_policy="zero")
aggregator.reset(series_length=17, labels=labels)
tensor_model = SimpleNamespace(score=lambda x: model.score(x)["fused"])
for start in (0, 4, 8, 11):
    aggregator.update({"ts_org": torch.from_numpy(test[None, start:start+6]),
                       "meta": {"start_idx": np.array([start])}}, tensor_model)
np.testing.assert_allclose(aggregator.result(finalize=True)["scores"], test[:, 0])

class BrokenModel(Model):
    def score(self, x):
        raise RuntimeError("intentional model failure")

model = BrokenModel()
try:
    evaluator(model, train, test, test_labels=labels)
except RuntimeError as error:
    assert str(error) == "intentional model failure"
else:
    raise AssertionError("model errors must propagate")
assert model.training, "training state must be restored after inference errors"

print("Configurable modes, direct aggregation, and failure restoration passed.")

for constant_label in (0, 1):
    evaluator = Evaluator({"wsz": 6, "stride": 4, "eval_window_size": 2})
    result = evaluator(Model(), train, test, test_labels=np.full(17, constant_label))
    for mode, values in result.items():
        assert values["calibrated"]["VUS_ROC"] is None
        assert values["oracle"]["f1_score"] == float(constant_label)
        if not constant_label:
            current = evaluator.aggregated[mode]["test"]
            assert not (current["scores"] >= values["oracle"]["threshold"]).any()

# YAML inheritance can disable a preset without needing to replace the mapping.
config = {"wsz": 6, "stride": 4, **deepcopy(presets)}
config["evaluator"]["aggregators"]["window"]["enabled"] = False
config["evaluator"]["aggregators"]["non_overlap_window"]["enabled"] = False
assert list(Evaluator(config).plans) == ["timeseries"]

for overrides in ({"calibration_quantile": 1.1}, {"batch_size": 0},
                  {"aggregators": {}}, {"metrics": "f1_score"},
                  {"aggregators": {"off": {"enabled": False}}}):
    try:
        Evaluator({"wsz": 6, "stride": 4, "evaluator": overrides})
    except ValueError:
        pass
    else:
        raise AssertionError(f"invalid evaluator settings accepted: {overrides}")

# A mode explicitly marked non-overlapping must reject overlapping scored spans.
config = {"wsz": 6, "stride": 4, "eval_window_size": 2, "evaluator": {
    "aggregators": {"bad_non_overlap": {**presets["evaluator"]["aggregators"]["non_overlap_window"],
                                       "stride": 1}}}}
model = Model()
try:
    Evaluator(config)(model, train, test, test_labels=labels)
except ValueError as error:
    assert "intervals overlap" in str(error)
else:
    raise AssertionError("overlapping non-overlap mode was accepted")
assert model.training
print("Single-class evaluation, disabled modes, configuration validation, and overlap rejection passed.")
