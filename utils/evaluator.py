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
        covered = counts > 0 if plan["aggregator"].per_window else np.ones_like(counts)
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
            check_dataset = plan.data_transform(dataset)
            dataloader = get_val_dataloader(check_dataset)
            for i, batch in enumerate(dataloader):
                score = plan.aggregator(model, batch)
                target = batch["target"]
                evaluation_metrics = combine_all_evaluation_scores(score, target, self.threshold[name], self.vus_window_size) #TODO: keep a running metrics evaluation and or scores for "live" view
                self.logger.metrics_summary(f"running/{name}", evaluation_metrics, i)