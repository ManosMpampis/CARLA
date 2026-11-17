import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
import yaml
from sklearn.metrics import precision_recall_curve

from metrics.metrics import combine_all_evaluation_scores
from utils.timeseries_aggregation import TimeseriesAggregator


def evaluation_options(config: dict) -> dict:
    """Resolve evaluation-only YAML options without changing training windows."""
    options = config.get("evaluation", {}) or {}
    if not isinstance(options, dict):
        raise ValueError("evaluation must be a YAML mapping")
    window = int(options.get("input_window", config["wsz"]))
    stride = int(options.get("stride", config["stride"]))
    return {
        "wsz": window,
        "stride": stride,
        "check_mask": options.get("check_mask"),
        "latency_offset": options.get("latency_offset", 0),
        # A final off-grid window would overlap even when stride == window.
        "include_last_window": options.get("include_last_window", stride < window),
    }

class Evaluator:
    """Score train/test windows once and evaluate configurable aggregations.

    ``config['evaluator']`` overrides the presets in configs/evaluator.yml.
    Supplying ``aggregators`` replaces the mode mapping, so its enabled entries
    select which modes run. Numeric aggregation fields also accept the named
    values documented in that YAML. Existing evaluation window/placement
    settings remain under ``config['evaluation']``.

    Call ``evaluate(model, train_dataset, test_dataset, epoch=...)`` with dataset
    objects exposing ``data`` or ``series`` and test ``targets``. Arrays are also
    accepted with ``test_labels=...``. Calibration uses ``full_train_series``
    when available. Corpus datasets must be evaluated one series at a time.

    Results are mode -> calibrated/oracle -> threshold and selected metrics.
    ``aggregators`` and ``aggregated`` expose independent train/test state and
    score/label/coverage snapshots. Optional logging uses the repo's Logger.
    Legacy function entry points below retain their existing reporting format.
    """

    def __init__(self, config, device=None, logger=None):
        self.config = deepcopy(config)
        with (Path(__file__).resolve().parents[1] / "configs/evaluator.yml").open() as stream:
            defaults = yaml.safe_load(stream)["evaluator"]
        settings = self.config.get("evaluator", {})
        if not isinstance(settings, dict):
            raise ValueError("evaluator must be a mapping")
        settings = {**defaults, **settings}
        self.options = evaluation_options(self.config)
        self.device = device if device is not None else self.config.get("device")
        self.logger = logger
        quantile = settings.get("calibration_quantile")
        if quantile is None:
            quantile = self.config.get("calibration_quantile", self.config.get(
                "calibration_kwargs", {}).get("quantile", 0.99))
        self.quantile = float(quantile)
        if not np.isfinite(self.quantile) or not 0 <= self.quantile <= 1:
            raise ValueError("calibration_quantile must be in [0, 1]")
        self.batch_size = settings.get("batch_size")
        if self.batch_size is None:
            self.batch_size = self.config.get("score_batch_size", self.config.get("batch_size", 256))
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError("evaluation batch_size must be a positive integer")
        self.metrics = settings["metrics"]
        if self.metrics is not None and (not isinstance(self.metrics, list) or
                                        any(not isinstance(key, str) for key in self.metrics)):
            raise ValueError("evaluator.metrics must be a list of metric names or null")
        specs = settings["aggregators"]
        if not isinstance(specs, dict) or not specs:
            raise ValueError("evaluator.aggregators must be a nonempty mapping")
        self.plans = {}
        for name, spec in specs.items():
            if not isinstance(name, str) or not name or not isinstance(spec, dict):
                raise ValueError("each aggregator needs a nonempty name and a parameter mapping")
            spec = dict(spec)
            enabled = spec.pop("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError("aggregator.enabled must be a boolean")
            if enabled:
                self.plans[name] = self._plan(spec)
        if not self.plans:
            raise ValueError("at least one evaluation aggregator must be enabled")
        self.aggregators = {}
        self.aggregated = {}
        self.results = {}

    def _plan(self, spec):
        params = dict(spec)
        metric_window = params.pop("metric_window_size", None)
        non_overlap = params.pop("require_non_overlap", False)
        window = params.get("input_window", "input_window")
        if window == "input_window":
            window = self.options["wsz"]
        span = self.options["check_mask"]
        if span is not None and (not isinstance(span, (list, tuple)) or len(span) != 2 or
                                 any(not isinstance(v, int) for v in span) or span[1] <= span[0]):
            raise ValueError("evaluation.check_mask must be a nonempty integer [start, end] interval")
        values = {
            "input_window": window,
            "evaluation_stride": self.options["stride"],
            "scored_span": window if span is None else span[1] - span[0],
            "output_start": (0 if span is None else span[0]) + self.options["latency_offset"],
            "evaluation_last_window": self.options["include_last_window"],
        }
        for key, default in (("input_window", "input_window"), ("stride", "evaluation_stride"),
                             ("time_offset", "output_start"), ("include_last_window", False)):
            value = params.get(key, default)
            if isinstance(value, str):
                if value not in values:
                    raise ValueError(f"unknown evaluator value {value!r} for {key}")
                value = values[value]
            params[key] = value
        params.setdefault("output_window", "output_length")
        params.setdefault("boundary_policy", "crop")
        params.setdefault("gap_policy", "zero")
        # Validate constructor values before running any inference.
        validated = {**params, "output_window": 1 if params["output_window"] == "output_length"
                     else params["output_window"]}
        TimeseriesAggregator(**validated)
        if metric_window is None:
            metric_window = int(self.config.get("eval_window_size", 100))
            if params.get("data_fn") == "window":
                metric_window = max(1, metric_window // params["stride"])
        if isinstance(metric_window, bool) or not isinstance(metric_window, int) or metric_window < 1:
            raise ValueError("metric_window_size must be a positive integer")
        if not isinstance(non_overlap, bool):
            raise ValueError("require_non_overlap must be a boolean")
        if non_overlap and (params.get("data_fn") != "window" or params["output_window"] != 1):
            raise ValueError("require_non_overlap needs scalar window aggregation")
        return {"params": params, "metric_window_size": metric_window,
                "require_non_overlap": non_overlap,
                "expected_length": values["scored_span"] if spec.get("time_offset", "output_start") == "output_start" else None}

    @staticmethod
    def _series(dataset, *, train=False):
        if getattr(dataset, "_corpus", None) is not None:
            raise ValueError("Evaluate each series separately, not a multi-series corpus")
        if isinstance(dataset, torch.Tensor):
            dataset = dataset.detach().cpu().numpy()
        if not isinstance(dataset, np.ndarray):
            attributes = ("full_train_series", "series", "data") if train else ("series", "data")
            for name in attributes:
                if hasattr(dataset, name):
                    dataset = getattr(dataset, name)
                    break
        series = np.asarray(dataset, dtype=np.float32)
        if series.ndim != 2 or not series.shape[1] or not np.isfinite(series).all():
            raise ValueError("evaluation data must be finite (timesteps, channels) arrays")
        return series

    @torch.no_grad()
    def evaluate(self, model, train_dataset, test_dataset, *, test_labels=None, epoch=0):
        train = self._series(train_dataset, train=True)
        test = self._series(test_dataset)
        if train.shape[1] != test.shape[1]:
            raise ValueError("train and test must have the same channel count")
        if test_labels is None:
            test_labels = getattr(test_dataset, "targets", None)
        labels = np.asarray(test_labels)
        if labels.ndim != 1 or len(labels) != len(test) or not np.isin(labels, [0, 1]).all():
            raise ValueError("test_labels must be binary timestep labels matching test data")
        self.aggregators = {name: {} for name in self.plans}
        self.aggregated = {name: {} for name in self.plans}
        self.results = {}
        self._score_lengths = {}
        device = self.device
        if device is None:
            device = next(iter(getattr(model, "parameters", lambda: iter(()))()), torch.empty(0)).device
        training = getattr(model, "training", None)
        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        try:
            self._score_split(model, train, None, "train", device)
            self._score_split(model, test, labels, "test", device)
        finally:
            if training is not None:
                model.train(training)
        for name, plan in self.plans.items():
            clean = self._covered(self.aggregated[name]["train"])
            target = self._covered(self.aggregated[name]["test"])
            calibrated = float(np.quantile(clean["scores"], self.quantile))
            oracle = self._oracle_threshold(target["scores"], target["labels"], inclusive=True)
            results = {}
            for source, threshold in (("calibrated", calibrated), ("oracle", oracle)):
                metrics = combine_all_evaluation_scores(
                    target["scores"], target["labels"], threshold,
                    window_size=plan["metric_window_size"])
                if self.metrics is not None:
                    unknown = set(self.metrics) - metrics.keys()
                    if unknown:
                        raise ValueError(f"unknown evaluation metrics: {sorted(unknown)}")
                    metrics = {key: metrics[key] for key in self.metrics}
                results[source] = {"threshold": threshold, **metrics}
                if source == "calibrated":
                    results[source]["calibration_quantile"] = self.quantile
            self.results[name] = results
        if self.logger is not None:
            self._log(epoch)
        return deepcopy(self.results)

    __call__ = evaluate

    def _score_split(self, model, series, labels, split, device):
        groups = {}
        for name, plan in self.plans.items():
            groups.setdefault(plan["params"]["input_window"], []).append(name)
        for window, names in groups.items():
            if len(series) < window:
                raise ValueError(f"{split} series length {len(series)} is shorter than input_window={window}")
            required = {}
            for name in names:
                params = self.plans[name]["params"]
                starts = set(range(0, len(series) - window + 1, params["stride"]))
                if params["include_last_window"]:
                    starts.add(len(series) - window)
                required[name] = starts
            all_starts = sorted(set().union(*required.values()))
            for begin in range(0, len(all_starts), self.batch_size):
                starts = np.asarray(all_starts[begin:begin + self.batch_size], dtype=np.int64)
                batch = np.stack([series[s:s + window] for s in starts])
                x = torch.from_numpy(batch).transpose(1, 2).contiguous().to(device)
                scores = model.score(x) if hasattr(model, "score") else model(x)
                if isinstance(scores, dict):
                    scores = scores["fused"]
                if isinstance(scores, torch.Tensor):
                    scores = scores.detach().float().cpu().numpy()
                scores = np.asarray(scores, dtype=np.float32)
                if scores.ndim != 2 or scores.shape[0] != len(starts) or not scores.shape[1] or not np.isfinite(scores).all():
                    raise ValueError("model must return finite (batch, output_length) scores")
                if self._score_lengths.setdefault(window, scores.shape[1]) != scores.shape[1]:
                    raise ValueError("model score length changed between batches or train/test splits")
                for name in names:
                    selected = np.asarray([int(s) in required[name] for s in starts])
                    if not selected.any():
                        continue
                    plan = self.plans[name]
                    expected = plan["expected_length"]
                    if expected is not None and scores.shape[1] != expected:
                        raise ValueError(f"{name}: score length {scores.shape[1]} differs from placed span {expected}")
                    if split not in self.aggregators[name]:
                        params = dict(plan["params"])
                        if params["output_window"] == "output_length":
                            params["output_window"] = scores.shape[1]
                        aggregator = TimeseriesAggregator(**params)
                        aggregator.reset(series_length=len(series), labels=labels)
                        self.aggregators[name][split] = aggregator
                    aggregator = self.aggregators[name][split]
                    aggregator.update_scores(scores[selected], starts[selected])
            for name in names:
                result = self.aggregators[name][split].result(finalize=True)
                covered = self._covered(result)
                if self.plans[name]["require_non_overlap"]:
                    intervals = covered["intervals"]
                    if np.any(intervals[1:, 0] < intervals[:-1, 1]):
                        raise ValueError(f"{name}: scored timestep intervals overlap")
                self.aggregated[name][split] = result

    @staticmethod
    def _covered(result):
        covered = result["cover_counts"] > 0
        if not covered.any():
            raise ValueError("evaluation has no covered positions")
        return {key: None if result[key] is None else result[key][covered]
                for key in ("scores", "labels", "intervals")}

    def _log(self, epoch):
        for mode, results in self.results.items():
            for source, values in results.items():
                phase = f"Test_{mode}/{source}"
                finite = {key: value for key, value in values.items()
                          if value is not None and np.isfinite(value)}
                if hasattr(self.logger, "metrics_summary"):
                    self.logger.metrics_summary(phase, finite, epoch)
                if hasattr(self.logger, "log"):
                    self.logger.log(f"{phase} epoch {epoch} | " + ", ".join(
                        f"{key}={value:.6f}" for key, value in finite.items()))


    @staticmethod
    def _oracle_threshold(scores: np.ndarray, labels: np.ndarray, *, inclusive=False) -> float:
        scores = np.asarray(scores, dtype=np.float32)
        if not np.any(labels):
            if inclusive:
                return float(np.nextafter(np.max(scores), np.float32(np.inf)))
            return float(np.max(scores))
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = np.divide(2 * precision[:-1] * recall[:-1],
                    precision[:-1] + recall[:-1],
                    out=np.zeros_like(precision[:-1]),
                    where=(precision[:-1] + recall[:-1]) > 0)
        # Predictions use strict >. Move one representable float below the
        # selected observed score so that the maximizing score remains included.
        if inclusive:
            return float(thresholds[int(np.argmax(f1))])
        return float(np.nextafter(thresholds[int(np.argmax(f1))], -np.inf))
