"""Check the original evaluator on timestep scores and window classification."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluator import Evaluator
from utils.timeseries_aggregation import TimeseriesAggregator


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def score(self, windows):
        assert not self.training
        self.seen.extend(windows[:, 0, 0].tolist())
        return {"fused": windows[:, 0]}


def check_window_classification():
    name = "windows"
    values = np.arange(12, dtype=np.float32) ** 2
    train = SimpleNamespace(data=values[:, None] / 10, targets=np.empty(0), wsz=4, stride=2)
    labels = np.zeros(12, dtype=int)
    labels[6] = 1
    test = SimpleNamespace(data=values[:, None], targets=labels, wsz=4, stride=2)
    aggregation = {"input_length": 4, "check_window": (0, 1), "mask": None,
                   "aggregate_scores_strategy": "mean"}
    for latency in (0, 2):
        starts = range(0, len(values) - latency - test.wsz + 1, test.stride)
        expected_scores = np.array([values[i:i + test.wsz].mean() for i in starts])
        expected_targets = np.array([labels[i + latency:i + latency + test.wsz].any() for i in starts])
        results = []
        for width in (1, test.stride):
            plan = {"latency": latency, "aggregator": {**aggregation, "check_window": (0, width)}}
            evaluator = Evaluator("cpu", None, {name: plan}, batch_size=2)
            model = Model()
            evaluator.calibrate(model, train, 0.9)
            result = evaluator(model, test)[name]
            results.append(result)
            wrapper = evaluator.plans[name]["data_wrapper"]
            aggregator = evaluator.plans[name]["aggregator"]
            timeline = np.zeros(starts[-1] + width)
            coverage = np.zeros_like(timeline)
            for start, score in zip(starts, expected_scores):
                timeline[start:start + width] = score
                coverage[start:start + width] = 1
            np.testing.assert_array_equal(aggregator.results(), timeline)
            np.testing.assert_array_equal(aggregator.counts, coverage)
            np.testing.assert_array_equal(wrapper.get_window_targets(), expected_targets)
            np.testing.assert_allclose(evaluator.train_scores[name], expected_scores / 10)
            oracle = evaluator.threshold[name]["oracle"]["threshold"]
            assert result["oracle"]["tp"] == np.sum((expected_scores >= oracle) & expected_targets)
            np.testing.assert_allclose(model.seen, [values[i] / 10 for i in starts] + [values[i] for i in starts])
        assert results[0] == results[1]

    masked = TimeseriesAggregator(3, (0, 4), (1, 3), aggregate_scores_strategy="mean")
    masked.update_record(torch.tensor([[30, 3, 6]]), torch.tensor([0]))
    np.testing.assert_array_equal(masked.results(), [3, 3, 3, 3])

    try:
        Evaluator("cpu", None, {name: {"latency": -1, "aggregator": aggregation}})
    except ValueError as exc:
        assert "latency" in str(exc)
    else:
        raise AssertionError("negative latency was accepted")


def main():
    name = "non-overlapping-windows"
    aggregation = {"input_length": 100, "check_window": (0, 100), "mask": None,
                   "aggregate_scores_strategy": None, "weighted_output": False}
    plan = {"eval_wsz": 100, "eval_stride": 100, "latency": 0,
            "true_target_policy": "count", "true_target_window": (0, 1),
            "true_target_threshold": 1, "aggregator": aggregation}
    evaluator = Evaluator("cpu", None, {name: plan})
    model = Model()
    values = np.arange(325, dtype=np.float32)
    train = SimpleNamespace(data=values[:, None] / 10, targets=np.empty(0), wsz=80, stride=5)
    test_labels = (values >= 200).astype(int)
    test = SimpleNamespace(data=values[:, None], targets=test_labels, wsz=80, stride=5)

    evaluator.calibrate(model, train, 0.9)
    result = evaluator(model, test)[name]
    assert model.seen == [0, 10, 20, 0, 100, 200]
    assert model.training
    assert plan["eval_stride"] == 100
    assert evaluator.threshold[name]["calibrate"]["quantile"] == 0.9
    assert evaluator.threshold[name]["oracle"]["threshold"] == 200
    assert result["oracle"]["f1_score"] == 1.0
    np.testing.assert_array_equal(evaluator.plans[name]["data_wrapper"].get_target_ts(), test_labels[:300])
    np.testing.assert_array_equal(evaluator.plans[name]["aggregator"].results(), values[:300])
    np.testing.assert_array_equal(evaluator.plans[name]["aggregator"].counts, np.ones(300))
    assert evaluator(model, test)[name] == result

    # A batch longer than the temporal window must still keep every score row.
    aggregator = TimeseriesAggregator(**aggregation)
    scores = torch.arange(10100, dtype=torch.float32)
    aggregator.update_record(scores.reshape(101, 100), torch.arange(0, 10100, 100))
    np.testing.assert_array_equal(aggregator.results(), scores.numpy())
    np.testing.assert_array_equal(aggregator.counts, np.ones(10100))

    cropped = TimeseriesAggregator(4, (1, 3), None)
    cropped.update_record(torch.tensor([[0, 2, 4, 0], [0, 8, 10, 0]]), torch.tensor([0, 1]))
    np.testing.assert_array_equal(cropped.results(), [0, 2, 6, 10])

    check_window_classification()
    print("Timestep scoring, overlap averaging, and window classification passed.")


if __name__ == "__main__":
    main()
