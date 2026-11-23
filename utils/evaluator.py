import torch
import numpy as np
import base64
import csv
import json
from copy import deepcopy
from pathlib import Path
import yaml
from sklearn.metrics import precision_recall_curve

from metrics.metrics import combine_all_evaluation_scores
from utils.timeseries_aggregation import TimeseriesAggregator
from utils.common_config import get_val_dataloader
from data.mapping_dataset import MappingDataset

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
    def __init__(self, device, logger, config: dict):
        self.vus_window_size = ...
        self.plans = {name: self._plan(plan) for name, plan in config.items()}
        self.threshold = {plan: {"calibrate": {"threshold":0, "quantile": 0}, "oracle": {"threshold":0, "quantile": 0}} for plan in config.keys()}
        pass

    def _plan(self, config):
        # Changes the data
        eval_config = {
            "latency": config.pop("latency", 0),
            "true_target_policy": config.pop("true_target_policy", "count"),
            "true_target_window": config.pop("true_target_window", (0, 1)),
            "true_target_value": config.pop("true_target_value", 1),
            "wsz": config.pop("eval_wsz", None),
            "stride": config.pop("eval_stride", None)
        }

        plan = {"aggregator": TimeseriesAggregator(**config["aggregator"]),
                "data_wrapper": MappingDataset(config["aggregator"], **eval_config) ,
            }
        return plan

    def calibrate(self, model, dataset, quantile):
        for name, plan in self.plans.items():
            plan.data_wrapper.reset() #TODO: find a way to not need to reset for the calibration and then again for the evaluation/__call__, because calibration runs with train data and evaluation with test, we need to recalculate the target ts if we reset
            plan.data_wrapper.assign_dataset(dataset)
            dataloader = get_val_dataloader(plan.data_wrapper)
            self.train_scores[name] = plan.aggregator.evaluate_ts(model, dataloader)
            self.threshold[name]["calibrate"]["threshold"] = float(np.quantile(self.train_scores[name], quantile))
            self.threshold[name]["calibrate"]["quantile"] = float(np.quantile(self.train_scores[name], quantile))

    def _oracle_threshold(self, scores: np.ndarray, targets: np.ndarray, name: str = None) -> float:
        precision, recall, thresholds = precision_recall_curve(targets, scores)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0, )
        oracle_th = float(thresholds[int(np.argmax(f1))])
        self.threshold[name]["oracle"]["threshold"] = oracle_th[name]
        if name:
            self.train_scores[name] = np.sort(self.train_scores[name])
            if oracle_th[name] <= self.train_scores[name][0]:
                self.threshold[name]["oracle"]["quantile"] = 0
                return
            elif oracle_th[name] >= self.train_scores[name][-1]:
                self.threshold[name]["oracle"]["quantile"] = 1
                return
            else:
                i = np.searchsorted(self.train_scores[name], oracle_th[name], side="left")
                if self.train_scores[name][i] == oracle_th[name]:
                    self.threshold[name]["oracle"]["quantile"] = i / (len(self.train_scores[name]) - 1) if len(self.train_scores[name]) > 1 else 0.0
                else:
                    fraction = (oracle_th[name] - self.train_scores[name][i - 1]) / (self.train_scores[name][i] - self.train_scores[name][i - 1])
                    self.threshold[name]["oracle"]["quantile"] = (i - 1 + fraction) / (len(self.train_scores[name]) - 1)
        return oracle_th
    
    def __call__(self, model, dataset):
        evaluation_metrics = {}
        for name, plan in self.plans.items():
            plan.data_wrapper.reset()
            plan.data_wrapper.assign_dataset(dataset)
            dataloader = get_val_dataloader(plan.data_wrapper)
            scores = plan.aggregator.evaluate_ts(model, dataloader)
            targets = plan.data_wrapper.get_target_ts()
            oracle_threshold = self._oracle_threshold(scores, targets, name)
            evaluation_metrics[name] = {
                "calibrated": combine_all_evaluation_scores(scores, targets, self.threshold[name]["calibrate"]["threshold"], self.vus_window_size),
                "oracle": combine_all_evaluation_scores(scores, targets, self.threshold[name]["oracle"]["threshold"], self.vus_window_size),
            }
        return evaluation_metrics

    def running_evaluator(self, model, dataset):
        for name, plan in self.plans.items():
            check_dataset = plan.data_transform(dataset)
            dataloader = get_val_dataloader(check_dataset)
            for batch in dataloader:
                score = plan.aggregator(model, batch)
                target = batch["target"]
                evaluation_metrics = combine_all_evaluation_scores(score, target, self.threshold[name], self.vus_window_size) #TODO: keep a running metrics evaluation and or scores for "live" view

class Evaluator2:
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
    ``aggregators`` contains separate train/test aggregators from construction.
    Each evaluation resets them and stores score/label/coverage snapshots in
    ``aggregated``. Optional logging uses the repo's Logger.
    Enable ``evaluator.artifacts`` or pass ``output_dir`` when evaluating to
    export CSV/HTML per mode and epoch. ``write_artifacts(directory)`` exports
    the latest results without inference. Exported paths live in ``artifacts``.
    """

    def __init__(self, config, device=None, logger=None):
        self.config = deepcopy(config)
        with (Path(__file__).resolve().parents[1] / "configs/evaluator.yml").open() as stream:
            defaults = yaml.safe_load(stream)["evaluator"]
        settings = self.config.get("evaluator", {})
        if not isinstance(settings, dict):
            raise ValueError("evaluator must be a mapping")
        settings = {**defaults, **settings}
        artifact_settings = settings["artifacts"]
        if not isinstance(artifact_settings, dict):
            raise ValueError("evaluator.artifacts must be a mapping")
        self.artifact_options = {**defaults["artifacts"], **artifact_settings}
        unknown = self.artifact_options.keys() - defaults["artifacts"].keys()
        if unknown:
            raise ValueError(f"unknown artifact settings: {sorted(unknown)}")
        for key in ("enabled", "csv", "html"):
            if not isinstance(self.artifact_options[key], bool):
                raise ValueError(f"evaluator.artifacts.{key} must be a boolean")
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
        self.aggregators = {}
        for name, spec in specs.items():
            if not isinstance(name, str) or not name or not isinstance(spec, dict):
                raise ValueError("each aggregator needs a nonempty name and a parameter mapping")
            spec = dict(spec)
            enabled = spec.pop("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError("aggregator.enabled must be a boolean")
            if enabled:
                plan = self._plan(spec)
                params = plan["params"]
                train_aggregator = TimeseriesAggregator(**params)
                if plan["metric_window_size"] is None:
                    metric_window = int(self.config.get("eval_window_size", 100))
                    if train_aggregator.data_fn == "window":
                        metric_window = max(1, metric_window // train_aggregator.stride)
                    if metric_window < 1:
                        raise ValueError("metric_window_size must be a positive integer")
                    plan["metric_window_size"] = metric_window
                self.plans[name] = plan
                self.aggregators[name] = {"train": train_aggregator, "test": TimeseriesAggregator(**params)}
        if not self.plans:
            raise ValueError("at least one evaluation aggregator must be enabled")
        self.aggregated = {}
        self.results = {}
        self.artifacts = {}

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
        if params.get("output_window", "output_length") == "output_length":
            params["output_window"] = None
        params.setdefault("boundary_policy", "crop")
        params.setdefault("gap_policy", "zero")
        if metric_window is not None and (isinstance(metric_window, bool) or
                                          not isinstance(metric_window, int) or metric_window < 1):
            raise ValueError("metric_window_size must be a positive integer")
        if not isinstance(non_overlap, bool):
            raise ValueError("require_non_overlap must be a boolean")
        if non_overlap and (params.get("data_fn") != "window" or params["output_window"] != 1):
            raise ValueError("require_non_overlap needs scalar window aggregation")
        return {"params": params, "metric_window_size": metric_window,
                "require_non_overlap": non_overlap,
                "expected_length": values["scored_span"] if spec.get("time_offset", "output_start") == "output_start" else None}

    @staticmethod
    def _series(dataset, train=False):
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
    def evaluate(self, model, train_dataset, test_dataset, *, test_labels=None, epoch=0,
                 output_dir=None):
        export = self.artifact_options["enabled"] or output_dir is not None
        export_root = self._artifact_root(output_dir, epoch) if export else None
        train = self._series(train_dataset, train=True)
        test = self._series(test_dataset)
        if train.shape[1] != test.shape[1]:
            raise ValueError("train and test must have the same channel count")
        if test_labels is None:
            test_labels = getattr(test_dataset, "targets", None)
        labels = np.asarray(test_labels)
        if labels.ndim != 1 or len(labels) != len(test) or not np.isin(labels, [0, 1]).all():
            raise ValueError("test_labels must be binary timestep labels matching test data")
        for aggregators in self.aggregators.values():
            aggregators["train"].reset(series_length=len(train))
            aggregators["test"].reset(series_length=len(test), labels=labels)
        self.aggregated = {name: {} for name in self.plans}
        self.results = {}
        self.artifacts = {}
        self._score_lengths = {}
        device = self.device
        if device is None:
            device = next(iter(getattr(model, "parameters", lambda: iter(()))()), torch.empty(0)).device

        self._score_split(model, train, "train", device)
        self._score_split(model, test, "test", device)
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
        if export:
            self.write_artifacts(export_root)
        return deepcopy(self.results)

    __call__ = evaluate

    def _artifact_root(self, directory, epoch):
        for name in self.plans:
            if name in (".", "..") or Path(name).name != name or "\\" in name or "\0" in name:
                raise ValueError("artifact mode names must be single directory names")
        if directory is None:
            directory = self.artifact_options["output_dir"]
        if directory is None:
            root = self.config.get("tag_dir") or self.config.get("jepa_dir")
            if root is None:
                raise ValueError("set evaluator.artifacts.output_dir or pass output_dir to export artifacts")
            directory = Path(root) / "evaluation"
        directory = Path(directory).expanduser().resolve()
        if epoch is not None:
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
                raise ValueError("artifact epoch must be a nonnegative integer or None")
            directory /= f"epoch_{epoch:06d}"
        return directory

    def write_artifacts(self, output_dir=None, *, epoch=None):
        """Export the latest evaluation, with no model calls or metric recomputation.

        Each mode directory contains scores.csv (covered test outputs, original
        half-open intervals, counts and both >= predictions), metrics.csv (one
        row per threshold source), and report.html. CSV/HTML can be disabled
        independently in evaluator.artifacts. Passing epoch adds an epoch
        directory; automatic exports use the evaluate() epoch.
        """
        if not self.results or self.results.keys() != self.plans.keys():
            raise ValueError("complete an evaluation before exporting artifacts")
        root = self._artifact_root(output_dir, epoch)
        paths = {}
        for name, metrics in self.results.items():
            result = self.aggregated[name]["test"]
            keep = result["cover_counts"] > 0
            values = {key: result[key][keep] for key in
                      ("scores", "labels", "indices", "intervals", "cover_counts")}
            predictions = {source: (values["scores"] >= values_["threshold"]).astype(np.uint8)
                           for source, values_ in metrics.items()}
            directory = root / name
            files = {}
            if self.artifact_options["csv"] or self.artifact_options["html"]:
                directory.mkdir(parents=True, exist_ok=True)
            if self.artifact_options["csv"]:
                path = directory / "scores.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("axis", "index", "start", "end", "score", "label", "cover_count",
                                     "calibrated_prediction", "oracle_prediction"))
                    for i, score in enumerate(values["scores"]):
                        start, end = values["intervals"][i]
                        writer.writerow((result["axis"], int(values["indices"][i]), int(start), int(end),
                                         float(score), int(values["labels"][i]), int(values["cover_counts"][i]),
                                         int(predictions["calibrated"][i]), int(predictions["oracle"][i])))
                files["scores_csv"] = str(path)
                path = directory / "metrics.csv"
                fields = list(dict.fromkeys(key for row in metrics.values() for key in row))
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["threshold_source", *fields])
                    writer.writeheader()
                    writer.writerows({"threshold_source": source, **row} for source, row in metrics.items())
                files["metrics_csv"] = str(path)
            if self.artifact_options["html"]:
                path = directory / "report.html"
                self._write_report(path, name, result["axis"], values, predictions, metrics)
                files["html"] = str(path)
            paths[name] = files
        self.artifacts = paths
        return deepcopy(paths)

    @staticmethod
    def _write_report(path, name, axis, values, predictions, metrics):
        def packed(array, dtype):
            return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode("ascii")

        def json_value(value):
            if isinstance(value, dict):
                return {key: json_value(item) for key, item in value.items()}
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                # Preserve infinite thresholds; undefined metrics are null.
                return None if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
            return value

        payload = {
            "name": name, "axis": axis, "prediction_operator": ">=",
            "metrics": json_value(metrics),
            "arrays": {
                "indices": packed(values["indices"], "<f8"),
                "scores": packed(values["scores"], "<f4"),
                "labels": packed(values["labels"], "u1"),
                "intervals": packed(values["intervals"], "<f8"),
                "counts": packed(values["cover_counts"], "<u4"),
                **{source: packed(prediction, "u1") for source, prediction in predictions.items()},
            },
        }
        encoded = base64.b64encode(json.dumps(payload, allow_nan=False).encode("utf-8")).decode("ascii")
        template = Path(__file__).with_name("evaluation_report.html").read_text(encoding="utf-8")
        path.write_text(template.replace("__EVALUATION_DATA__", encoded), encoding="utf-8")

    def _score_split(self, model, series, split, device):
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
