"""Shared anomaly evaluation and thin native-training compatibility adapters."""

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from data.jepa_dataset import JEPADataset
from utils.common_config import get_jepa_datasets
from utils.config import create_config, entry_overrides, load_experiment_config
from utils.evaluation_metrics import decision_metrics, score_metrics
from utils.evaluation_logging import log_evaluation, record_evaluation
from utils.scoring import (aggregate_score_maps, covered_evaluation_view,
                           evaluation_options)


@torch.no_grad()
def score_both(model, series: np.ndarray, window: int, stride: int,
               batch_size: int, device, *, check_mask=None,
               latency_offset: int = 0,
               include_last_window: bool | None = None) -> dict:
    """Window scalar = mean of score map; series = cover-count mean of maps."""
    model.eval()
    def score_batch(windows):
        x = torch.from_numpy(windows).permute(0, 2, 1).contiguous().to(device)
        return {"fused": model.score(x)["fused"].float().cpu().numpy()}

    result = aggregate_score_maps(
        series, window, stride, batch_size, score_batch,
        check_mask=check_mask, latency_offset=latency_offset,
        include_last_window=include_last_window)
    return {"window_scores": result["window_scores"]["fused"],
            "timeseries_scores": result["channels"]["fused"],
            "starts": result["start_idxs"], "ends": result["end_idxs"],
            "input_starts": result["input_start_idxs"],
            "input_ends": result["input_end_idxs"],
            "cover_counts": result["cover_counts"]}


def _oracle_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if not np.any(labels):
        return float(np.max(scores))
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(2 * precision[:-1] * recall[:-1],
                   precision[:-1] + recall[:-1],
                   out=np.zeros_like(precision[:-1]),
                   where=(precision[:-1] + recall[:-1]) > 0)
    # Predictions use strict >. Move one representable float below the
    # selected observed score so that the maximizing score remains included.
    return float(np.nextafter(thresholds[int(np.argmax(f1))], -np.inf))


def _classification(scores, labels, threshold) -> dict:
    return decision_metrics(scores, labels, threshold)


def calibration_at_threshold(clean_scores, threshold) -> dict:
    """Invert NumPy's linear quantile; clamp thresholds outside its score range.

    The empirical anomaly fraction uses the same strict > as predictions.
    It differs from 1-q because quantiles interpolate between order statistics.
    """
    scores = np.sort(np.asarray(clean_scores, dtype=np.float64).reshape(-1))
    if not scores.size or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("calibration diagnostics require finite, nonempty scores and threshold")
    index = int(np.searchsorted(scores, threshold, side="right"))
    if scores.size == 1 or index == 0:
        quantile = 0.0
    elif index == scores.size:
        quantile = 1.0
    else:
        lower, upper = scores[index - 1], scores[index]
        quantile = float((index - 1 + (threshold - lower) / (upper - lower))
                         / (scores.size - 1))
    reproduced = float(np.quantile(scores, quantile))
    return {
        "calibration_quantile": quantile,
        "calibration_threshold": reproduced,
        "calibration_threshold_in_range": bool(scores[0] <= threshold <= scores[-1]),
        "clean_anomaly_fraction": float(np.mean(scores > threshold)),
    }


def _evaluate_protocol(scores, labels, clean_scores, quantile, vus_window):
    calibrated = float(np.quantile(clean_scores, quantile))
    oracle = _oracle_threshold(scores, labels)
    independent = score_metrics(scores, labels, vus_window)
    calibrated_metrics = _classification(scores, labels, calibrated)
    calibrated_metrics["calibration_quantile"] = float(quantile)
    oracle_metrics = _classification(scores, labels, oracle)
    oracle_metrics.update(calibration_at_threshold(clean_scores, oracle))
    for metrics in (calibrated_metrics, oracle_metrics):
        metrics["pa_f1"] = metrics.pop("pa_f1")
    return {
        "calibrated": calibrated_metrics,
        "oracle": oracle_metrics,
        **independent,
    }


def aggregation_modes(config):
    """Resolve the public aggregation names into the report keys."""
    names = ("window", "non_overlap_window", "timeseries")
    selection = config.get("aggregate_output_strategy",
                           config.get("agregate_output_strategy", "all"))
    if selection == "all":
        return names
    if isinstance(selection, str):
        selection = [selection]
    if not isinstance(selection, (list, tuple)) or not selection:
        raise ValueError("aggregate_output_strategy must be all or one or more aggregation names")
    modes = tuple("non_overlap_window" if mode == "non_overlap" else mode for mode in selection)
    if any(mode not in names for mode in modes):
        raise ValueError("aggregation names are window, non_overlap, and timeseries")
    return modes


def evaluate_from_scores(clean, test, test_labels, config, *, modes=("window", "timeseries")) -> dict:
    """Evaluate the selected score arrays with distinct thresholds."""
    stride = evaluation_options(config)["stride"]
    quantile = float(config.get("calibration_quantile", 0.99))
    vus_steps = int(config.get("eval_window_size", 100))
    result = {}
    if "window" in modes:
        labels = np.asarray([np.any(test_labels[s:e]) for s, e in zip(test["starts"], test["ends"])],
                            dtype=np.int64)
        result["window"] = _evaluate_protocol(test["window_scores"], labels, clean["window_scores"],
                                              quantile, max(1, vus_steps // stride))
    if "timeseries" in modes:
        clean_series = clean["timeseries_scores"][clean["cover_counts"] > 0]
        test_series, labels, _, _ = covered_evaluation_view(
            test["timeseries_scores"], test_labels, test["starts"],
            test["ends"], test["cover_counts"])
        result["timeseries"] = _evaluate_protocol(test_series, labels, clean_series, quantile, vus_steps)
    return result


def evaluate_aggregations(model, train_series, test_series, test_labels, config, device, *, options):
    """Evaluate only the requested aggregations and retain score arrays for export."""
    modes = aggregation_modes(config)
    options = dict(options)
    window, stride = options.pop("wsz"), options.pop("stride")
    metrics = {}
    test = non_overlap = non_overlap_stride = None
    if "window" in modes or "timeseries" in modes:
        batch_size = int(config.get("score_batch_size", config["batch_size"]))
        clean = score_both(model, train_series, window, stride, batch_size, device, **options)
        test = score_both(model, test_series, window, stride, batch_size, device, **options)
        metrics.update(evaluate_from_scores(clean, test, test_labels, config, modes=modes))
    if "non_overlap_window" in modes:
        metrics["non_overlap_window"], _, non_overlap, non_overlap_stride = evaluate_non_overlap_window(
            model, train_series, test_series, test_labels, config, device,
            options={"wsz": window, "stride": stride, **options})
    return metrics, test if test is not None else non_overlap, non_overlap, non_overlap_stride


def evaluate(model, train_series, test_series, test_labels, config, device, *, options=None) -> dict:
    if options is None:
        options = evaluation_options(config)
    metrics, _, _, _ = evaluate_aggregations(model, train_series, test_series, test_labels,
                                             config, device, options=options)
    return metrics


def evaluate_non_overlap_window(model, train_series, test_series, test_labels,
                                config, device, *, options):
    """Window-level metrics on disjoint scored regions using the same scorer.

    Input context may overlap for crop heads. Scored timesteps never overlap;
    initial context and any incomplete final scored region remain unscored.
    """
    options = dict(options)
    window = int(options.pop("wsz"))
    options.pop("stride", None)
    span = options.get("check_mask")
    stride = int(span[1] - span[0]) if span is not None else window
    options["include_last_window"] = False
    batch_size = int(config.get("score_batch_size", config["batch_size"]))
    clean = score_both(model, train_series, window, stride, batch_size, device, **options)
    test = score_both(model, test_series, window, stride, batch_size, device, **options)
    if np.any(clean["cover_counts"] > 1) or np.any(test["cover_counts"] > 1):
        raise ValueError("non_overlap_window scored a timestep more than once")
    labels = np.asarray([np.any(test_labels[s:e])
                         for s, e in zip(test["starts"], test["ends"])], dtype=np.int64)
    metrics = _evaluate_protocol(
        test["window_scores"], labels, clean["window_scores"],
        float(config.get("calibration_quantile", 0.99)),
        max(1, int(config.get("eval_window_size", 100)) // stride))
    return metrics, clean, test, stride


# Compatibility exports for callers; reporting itself has one implementation.
_log_evaluation = log_evaluation
record_test_evaluation = record_evaluation


def evaluate_on_test(model, config, device, logger, epoch, *, options=None,
                     input_resolution=False, datasets=None):
    """Monitoring hook shared by all LEWM stages and phase-two heads."""
    if datasets is None:
        train_dataset, _ = get_jepa_datasets(config)
        from data.jepa_dataset import JEPACorpusDataset
        test_dataset = None if isinstance(train_dataset, JEPACorpusDataset) else JEPADataset(config, train=False)
    else:
        train_dataset, test_dataset = datasets
    from utils.run_evaluation import evaluate_run
    report = evaluate_run(model, config, device, datasets=(train_dataset, test_dataset), write=False)
    model._last_evaluation_report = report
    metrics = report["evaluation"]

    class Writer:
        def add_scalar(self, tag, value, step):
            section, name = tag.rsplit("/", 1)
            logger.scalar_summary(section, name, value, step)

    _log_evaluation(Writer(), metrics, epoch)
    record_test_evaluation(config, metrics, epoch, logger.log)
    return metrics


def run_arm(arm, args, update_dictionary=None):
    """Single AE/VAE entry: train normally, or score the same saved experiment."""
    overrides = entry_overrides(args, update_dictionary)
    stage = overrides.get("stage", load_experiment_config(args.config_exp).get("stage", "pretrain"))
    if str(stage).lower() == "score":
        config = create_config(args.config_env, args.config_exp, args.fname,
                               args.version, update_dictionary=overrides)
        return score_arm(arm, args, config)
    return train_arm(arm, args, overrides)


def score_arm(arm, args, config):
    from utils.experiment_suite import score_config
    if arm not in ("ae", "vae"):
        raise ValueError(f"unknown reconstruction arm {arm}")
    return score_config(config, arm)


def train_arm(arm, args, update_dictionary=None):
    """Compatibility adapter; all optimization belongs to the native trainer."""
    from train import run
    config = create_config(args.config_env, args.config_exp, args.fname,
                           args.version, update_dictionary=update_dictionary or {})
    if arm not in ("ae", "vae") or config.get("framework") != arm:
        raise ValueError(f"config framework must match {arm}")
    return run(config)
