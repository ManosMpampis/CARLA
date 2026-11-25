"""Check the original evaluator on timestep scores and window classification."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.evaluator import Evaluator
from utils.timeseries_aggregation import TimeseriesAggregator
from data.mapping_dataset import MappingDataset


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
                   "aggregate_scores_strategy": "mean", "per_window": True}
    for latency in (0, 2):
        starts = range(0, len(values) - latency - test.wsz + 1, test.stride)
        expected_scores = np.array([values[i:i + test.wsz].mean() for i in starts])
        expected_targets = np.array([labels[i + latency:i + latency + test.wsz].any() for i in starts])
        results = []
        for width in (1, test.stride):
            plan = {"latency": latency, "aggregator": {**aggregation, "check_window": (0, width)}}
            evaluator = Evaluator("cpu", None, {"batch_size": 2, name: plan})
            model = Model()
            evaluator.calibrate(model, train, 0.9)
            result = evaluator(model, test)[name]
            results.append(result)
            wrapper = evaluator.plans[name]["data_wrapper"]
            aggregator = evaluator.plans[name]["aggregator"]
            np.testing.assert_array_equal(aggregator.results(), expected_scores)
            np.testing.assert_array_equal(aggregator.counts, np.ones(len(starts)))
            np.testing.assert_array_equal(wrapper.get_target_ts(), expected_targets)
            np.testing.assert_allclose(evaluator.train_scores[name], expected_scores / 10)
            oracle = evaluator.threshold[name]["oracle"]["threshold"]
            assert result["oracle"]["tp"] == np.sum((expected_scores >= oracle) & expected_targets)
            np.testing.assert_allclose(model.seen, [values[i] / 10 for i in starts] + [values[i] for i in starts])
        assert results[0] == results[1]

    masked = TimeseriesAggregator(4, (0, 4), (1, 3))
    masked.update_record(torch.tensor([[30, 3, 6, 30]]), torch.tensor([0]))
    np.testing.assert_array_equal(masked.results(), [0, 3, 6, 0])

    try:
        Evaluator("cpu", None, {name: {"latency": -1, "aggregator": aggregation}})
    except ValueError as exc:
        assert "latency" in str(exc)
    else:
        raise AssertionError("negative latency was accepted")


def check_target_cache():
    config = {"input_length": 4, "check_window": (1, 3), "mask": None}
    labels = np.array([0, 0, 1, 0, 0, 1, 1, 0, 0])
    series = SimpleNamespace(data=np.zeros((len(labels), 1)), targets=labels, wsz=4, stride=2)
    dataset = MappingDataset(config, latency=1, true_target_window=(-1, 2),
                             true_target_policy="percent", true_target_threshold=0.5)
    dataset.assign_dataset(series)
    np.testing.assert_array_equal(dataset[2]["target"], labels[5:9])
    np.testing.assert_array_equal(dataset[0]["target"], labels[1:5])
    assert len(dataset.targets_aggregator) == 0

    mapped = np.array([0, *labels[2:8]])
    expected = [np.mean(mapped[max(0, i - 1):i + 2]) >= 0.5 for i in range(len(mapped))]
    dataset.get_target_ts()[:] = 0
    np.testing.assert_array_equal(dataset.get_target_ts(), expected)
    counts = dataset.targets_aggregator.counts.copy()
    dataset[2]
    np.testing.assert_array_equal(dataset.get_target_ts(), expected)
    np.testing.assert_array_equal(dataset.targets_aggregator.counts, counts)
    dataset.reset()
    np.testing.assert_array_equal(dataset.get_target_ts(), expected)

    dataset.assign_dataset(SimpleNamespace(data=series.data, targets=[], wsz=4, stride=2))
    np.testing.assert_array_equal(dataset.get_target_ts(), np.zeros_like(expected))
    dataset.assign_dataset(SimpleNamespace(data=series.data[:1], targets=labels[:1], wsz=4, stride=2))
    assert len(dataset) == len(dataset.get_target_ts()) == 0


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
    check_target_cache()
    print("Timestep scoring, overlap averaging, window classification, and target caching passed.")


if __name__ == "__main__":
    main()
