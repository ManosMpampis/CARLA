import torch
import numpy as np
from sklearn.metrics import precision_recall_curve

from utils.evaluation_logging import log_evaluation, record_evaluation
from utils.scoring import evaluation_options, aggregate_score_maps, covered_evaluation_view
from utils.common_config import get_test_dataset, get_val_dataloader
from utils.evaluation_metrics import score_metrics, decision_metrics

# TODO: Make evaluator
# Run train dataset through model
# Run test dataset through model
# For any aggrigator do:
#   Apply aggrigator
#   Find calibrated threshold
#   Find oracle threshold
#   Output non threshold metrics
#   Output threshold metrics with calibrated th
#   Output threshold metrics with oracle th
#
#   Apply PA transformation to the test results
#   Output non threshold metrics
#   Output threshold metrics with calibrated th
#   Output threshold metrics with oracle th

# TODO: Make aggregator
# TODO: Make prefix configurations for the aggregator

class Evaluator:
    def __init__(self, train_dataset, config, device, logger, options=None):
        self.train_dataset = train_dataset
        self.test_dataset = get_test_dataset(config, train_dataset)
        self.train_dataloader = get_val_dataloader(config, self.train_dataset)
        self.test_dataloader = get_val_dataloader(config, self.test_dataset)

        self.device = device
        self.logger = logger

        self.calibratoin_quantile = float(config.get(
                "calibration_quantile", config.get("calibration_kwargs", {}).get("quantile", 0.99)))
        self.batch_size = int(config.get("score_batch_size", config["batch_size"])) #TODO: check for the batch size during config init
        self.window_size = int(config.get("eval_window_size", config["wsz"])) #TODO: check for the window size during config init
        self.stride = int(config.get("stride")) #TODO: check stride


def evaluate_on_test(model, train_dataset, config, device, logger, epoch, *, options=None):
    """Monitoring hook shared by all LEWM stages and phase-two heads."""
    test_dataset = get_test_dataset(config, train_dataset)
    eval_config = dict(config)
    eval_config["calibration_quantile"] = float(config.get(
        "calibration_quantile", config.get("calibration_kwargs", {}).get("quantile", 0.99)))

    metrics = evaluate(model, train_dataset.full_train_series, test_dataset.series,
                       test_dataset.targets, eval_config, device, options=options)

    log_evaluation(metrics, epoch)
    record_evaluation(config, metrics, epoch, logger.log)
    return metrics

def evaluate(model, train_series, test_series, test_labels, config, device,
             *, options=None) -> dict:
    """Distinct window and overlap-averaged time-series evaluation."""
    options = dict(evaluation_options(config) if options is None else options)
    window, stride = options.pop("wsz"), options.pop("stride")
    batch_size = int(config.get("score_batch_size", config["batch_size"]))
    clean = score_both(model, train_series, window, stride, batch_size, device,
                       **options)
    test = score_both(model, test_series, window, stride, batch_size, device,
                      **options)
    result = evaluate_from_scores(clean, test, test_labels, config)
    result["non_overlap_window"], _, _, _ = evaluate_non_overlap_window(
        model, train_series, test_series, test_labels, config, device,
        options={"wsz": window, "stride": stride, **options})
    return result

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
                         for s, e in zip(test["starts"], test["ends"])], dtype=np.int32)
    metrics = _evaluate_protocol(
        test["window_scores"], labels, clean["window_scores"],
        float(config.get("calibration_quantile", 0.99)),
        max(1, int(config.get("eval_window_size", 100)) // stride))
    return metrics, clean, test, stride

def _evaluate_protocol(scores, labels, clean_scores, quantile, vus_window):
    calibrated = float(np.quantile(clean_scores, quantile))
    oracle = _oracle_threshold(scores, labels)
    independent = score_metrics(scores, labels, vus_window)
    calibrated_metrics = decision_metrics(scores, labels, calibrated)
    calibrated_metrics["calibration_quantile"] = float(quantile)
    oracle_metrics = decision_metrics(scores, labels, oracle)
    oracle_metrics.update(calibration_at_threshold(clean_scores, oracle))
    for metrics in (calibrated_metrics, oracle_metrics):
        metrics["pa_f1"] = metrics.pop("pa_f1")
    return {
        "calibrated": calibrated_metrics,
        "oracle": oracle_metrics,
        **independent,
    }

def _oracle_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float32)
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

def calibration_at_threshold(clean_scores, threshold) -> dict:
    """Invert NumPy's linear quantile; clamp thresholds outside its score range.

    The empirical anomaly fraction uses the same strict > as predictions.
    It differs from 1-q because quantiles interpolate between order statistics.
    """
    scores = np.sort(np.asarray(clean_scores, dtype=np.float32).reshape(-1))
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

def evaluate_from_scores(clean, test, test_labels, config) -> dict:
    """Evaluate window and time-series score arrays with distinct thresholds."""
    stride = evaluation_options(config)["stride"]
    window_labels = np.asarray([
        np.any(test_labels[s:e]) for s, e in zip(test["starts"], test["ends"])
    ], dtype=np.int32)
    quantile = float(config.get("calibration_quantile", 0.99))
    vus_steps = int(config.get("eval_window_size", 100))
    clean_series = clean["timeseries_scores"][clean["cover_counts"] > 0]
    test_series, covered_labels, _, _ = covered_evaluation_view(
        test["timeseries_scores"], test_labels, test["starts"],
        test["ends"], test["cover_counts"])
    result = {
        "window": _evaluate_protocol(test["window_scores"], window_labels,
                                     clean["window_scores"], quantile,
                                     max(1, vus_steps // stride)),
        "timeseries": _evaluate_protocol(test_series,
                                         covered_labels,
                                         clean_series, quantile,
                                         vus_steps),
    }
    return result

@torch.no_grad()
def score_both(model, series: np.ndarray, window: int, stride: int,
               batch_size: int, device, *, check_mask=None,
               latency_offset: int = 0,
               include_last_window: bool | None = None) -> dict:
    """Window scalar = mean of score map; series = cover-count mean of maps."""
    model.eval()
    def score_batch(windows):
        x = torch.from_numpy(windows).permute(0, 2, 1).contiguous().to(device)
        return {"fused": model.score(x)["fused"].cpu().numpy()}

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