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
        self.logger = logger
        self.device = device
        self.loader_config = {"batch_size": config.pop("batch_size", 1), "num_workers": 0}
        self.vus_window_size = config.pop("vus_window_size",100)
        self.plans = {name: self._plan(plan) for name, plan in config.items()}
        self.train_scores = {}
        self.threshold = {
            name: {source: {"threshold": 0, "quantile": 0} for source in ("calibrate", "oracle")}
            for name in self.plans
        }

    def _plan(self, config):
        eval_config = {
            "latency": config.get("latency", 0),
            "true_target_policy": config.get("true_target_policy", "count"),
            "true_target_window": config.get("true_target_window", (0, 1)),
            "true_target_threshold": config.get("true_target_threshold", 1),
            "wsz": config.get("eval_wsz"),
            "stride": config.get("eval_stride"),
        }
        aggregator_config = {"device": self.device, **config["aggregator"]}
        return {
            "aggregator": TimeseriesAggregator(**aggregator_config),
            "data_wrapper": MappingDataset(aggregator_config, **eval_config),
        }

    def _score_plan(self, model, dataset, plan):
        plan["data_wrapper"].assign_dataset(dataset)
        dataloader = get_val_dataloader(self.loader_config, plan["data_wrapper"])
        scores = plan["aggregator"].evaluate_ts(model, dataloader)
        counts = plan["aggregator"].counts
        covered = counts > 0 if plan["aggregator"].per_window else np.ones_like(counts, dtype=bool)
        return scores[covered], covered
    
    def calibrate(self, model, dataset, quantile):
        for name, plan in self.plans.items():
            scores, _ = self._score_plan(model, dataset, plan)
            threshold = float(np.quantile(scores, quantile))
            self.train_scores[name] = np.sort(scores)
            self.threshold[name]["calibrate"] = {"threshold": threshold, "quantile": float(quantile)}

    def _oracle_threshold(self, scores: np.ndarray, targets: np.ndarray, name: str = None) -> float:
        if np.any(targets):
            precision, recall, thresholds = precision_recall_curve(targets, scores)
            precision, recall = precision[:-1], recall[:-1]
            total = precision + recall
            f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)
            threshold = float(thresholds[np.argmax(f1)])
        else:
            threshold = float(np.nextafter(scores.max(), np.inf))
        if name is not None:
            train_scores = self.train_scores.get(name)
            quantile = 0.0
            if train_scores is not None:
                ranks = np.linspace(0, 1, len(train_scores))
                quantile = float(np.interp(threshold, train_scores, ranks))
            self.threshold[name]["oracle"] = {"threshold": threshold, "quantile": quantile}
        return threshold

    def __call__(self, model, dataset):
        evaluation_metrics = {}
        for name, plan in self.plans.items():
            scores, _ = self._score_plan(model, dataset, plan)
            targets = plan["data_wrapper"].get_target_ts()

            evaluation_metrics[name] = {
                "calibrated": combine_all_evaluation_scores(scores, targets, self.threshold[name]["calibrate"]["threshold"], self.vus_window_size),
                "oracle": combine_all_evaluation_scores(scores, targets, self._oracle_threshold(scores, targets, name), self.vus_window_size),
            }
        
        return evaluation_metrics

    def running_evaluator(self, model, dataset):
        for name, plan in self.plans.items():
            plan.aggregator.reset()
            check_dataset = plan.data_transform(dataset)
            dataloader = get_val_dataloader(check_dataset)
            for i, batch in enumerate(dataloader):
                score = plan.aggregator(model, batch) #TODO: think what we need for running score, maybe check only the self.width output
                target = batch["target"]
                evaluation_metrics = combine_all_evaluation_scores(score, target, self.threshold[name], self.vus_window_size) #TODO: keep a running metrics evaluation and or scores for "live" view, what else need to be logged
                self.logger.metrics_summary(f"running/{name}/calibrated", evaluation_metrics, i)

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

    def write_artifacts(self, output_dir=None, epoch=None):
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
