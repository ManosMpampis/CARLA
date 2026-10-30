"""AE/VAE training and shared three-mode anomaly evaluation."""

import os
import random

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

from data.jepa_dataset import JEPADataset
from models.ae_baseline import reconstruction_objective
from models.builders import get_ae_model, get_vae_model
from utils.common_config import (get_jepa_datasets, get_optimizer, get_scheduler,
                                 get_train_dataloader, get_val_dataloader)
from utils.config import create_config, entry_overrides, load_experiment_config
from utils.evaluation_schedule import evaluation_interval, should_evaluate
from utils.checkpointing import Checkpoints, inference_metadata
from utils.utils import Logger
from utils.evaluation_metrics import decision_metrics, score_metrics
from utils.evaluation_logging import (log_evaluation, record_evaluation, loss_metrics,
                                      log_losses, record_losses, latent_variance, truncate_epoch_history)
from utils.scoring import (aggregate_score_maps, covered_evaluation_view,
                           evaluation_options)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(config):
    requested = str(config.get("device", "cpu"))
    return torch.device(requested if not requested.startswith("cuda") or
                        torch.cuda.is_available() else "cpu")


def _starts(length: int, window: int, stride: int) -> np.ndarray:
    if length < window:
        raise ValueError(f"series length {length} is shorter than window {window}")
    if stride < 1:
        raise ValueError("stride must be positive")
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return np.asarray(starts, dtype=np.int64)


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


def evaluate_from_scores(clean, test, test_labels, config) -> dict:
    """Evaluate window and time-series score arrays with distinct thresholds."""
    stride = evaluation_options(config)["stride"]
    window_labels = np.asarray([
        np.any(test_labels[s:e]) for s, e in zip(test["starts"], test["ends"])
    ], dtype=np.int64)
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


def evaluate(model, val_series, test_series, test_labels, config, device,
             *, options=None) -> dict:
    """Distinct window and overlap-averaged time-series evaluation."""
    options = dict(evaluation_options(config) if options is None else options)
    window, stride = options.pop("wsz"), options.pop("stride")
    batch_size = int(config.get("score_batch_size", config["batch_size"]))
    clean = score_both(model, val_series, window, stride, batch_size, device,
                       **options)
    test = score_both(model, test_series, window, stride, batch_size, device,
                      **options)
    result = evaluate_from_scores(clean, test, test_labels, config)
    result["non_overlap_window"], _, _, _ = evaluate_non_overlap_window(
        model, val_series, test_series, test_labels, config, device,
        options={"wsz": window, "stride": stride, **options})
    return result


def evaluate_non_overlap_window(model, val_series, test_series, test_labels,
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
    clean = score_both(model, val_series, window, stride, batch_size, device, **options)
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


def _run_epoch(model, loader, optimizer, device, beta: float, writer,
               epoch: int, training: bool) -> dict:
    model.train(training)
    totals = {}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            x = batch["ts"].float().transpose(1, 2).contiguous().to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(x, sample=training and model.variational)
            losses = reconstruction_objective(outputs, x, beta)
            if training:
                losses["loss"].backward()
                optimizer.step()
            latent = outputs["latent"].detach().float()
            losses = loss_metrics(losses, reconstruction_key="mse")
            losses["latent_var"] = latent_variance(latent)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * x.size(0)
            count += x.size(0)
    if count == 0:
        raise ValueError("empty data loader; lower batch size or window length")
    averages = {name: value / count for name, value in totals.items()}
    prefix = "train" if training else "validation"
    log_losses(writer, prefix, averages, epoch)
    return averages


# Compatibility exports for callers; reporting itself has one implementation.
_log_evaluation = log_evaluation
record_test_evaluation = record_evaluation


def evaluate_on_test(model, config, device, logger, epoch, *, options=None,
                     input_resolution=False):
    """Monitoring hook shared by all LEWM stages and phase-two heads."""
    _, val_dataset = get_jepa_datasets(config)
    test_dataset = JEPADataset(config, train=False)
    eval_config = dict(config)
    eval_config["calibration_quantile"] = float(config.get(
        "calibration_quantile", config.get("calibration_kwargs", {}).get("quantile", 0.99)))

    class InputResolutionScores:
        def eval(self):
            model.eval()
            return self

        def score(self, x):
            return {"fused": model.score(x)["fused"].repeat_interleave(
                int(model.level_strides[0]), dim=-1)}

    scorer = InputResolutionScores() if input_resolution else model
    metrics = evaluate(scorer, val_dataset.series, test_dataset.series,
                       test_dataset.targets, eval_config, device, options=options)

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
    """Evaluate validation-selected weights without creating an optimizer."""
    from pathlib import Path
    from utils.experiment_suite import Experiment, score_experiment

    if arm not in {"ae", "vae"} or config.get("arm") != arm:
        raise ValueError(f"config arm {config.get('arm')} does not match {arm}")
    _seed(int(config.get("seed", 4)))
    experiment = Experiment(config.get("framework", arm), config.get("experiment_name", "default"),
                            arm, Path(args.config_exp), dict(config))
    return score_experiment(experiment, args.config_env, config["version"], fname=args.fname)


def train_arm(arm: str, args, update_dictionary=None) -> dict:
    if arm not in {"ae", "vae"}:
        raise ValueError(f"unknown reconstruction arm {arm}")
    config = create_config(args.config_env, args.config_exp, args.fname,
                           args.version, update_dictionary=update_dictionary or {})
    if config.get("arm") != arm:
        raise ValueError(f"config arm {config.get('arm')} does not match {arm}")
    _seed(int(config.get("seed", 4)))
    device = _device(config)
    builder = get_vae_model if arm == "vae" else get_ae_model
    model = builder(config).to(device)
    if int(config["wsz"]) % model.total_stride:
        raise ValueError("window length must be divisible by encoder stride")
    train_dataset, val_dataset = get_jepa_datasets(config)
    if not isinstance(train_dataset, JEPADataset) or train_dataset._corpus is not None:
        raise ValueError("these comparison arms require a single machine")
    test_dataset = JEPADataset(config, train=False)
    train_loader = get_train_dataloader(config, train_dataset)
    val_loader = get_val_dataloader(config, val_dataset)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    out_dir = config["jepa_dir"]
    logger = Logger(config["version"], verbose=3, file_path=out_dir, delete_files=False)
    beta = float(config.get("beta", 0.3)) if arm == "vae" else 0.0
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    eval_every = evaluation_interval(config)
    last_path = config["jepa_checkpoint"]
    if not os.path.exists(last_path):
        last_path = os.path.join(out_dir, "last.pth.tar")  # legacy AE/VAE resume
    first_epoch, best = 1, None
    if os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best = checkpoint.get("best")
        first_epoch = int(checkpoint["epoch"]) + 1
    if first_epoch > 1:
        logger.resume_from(first_epoch - 1)
        truncate_epoch_history(out_dir, first_epoch - 1)
    writer = logger.experiment
    checkpoints = Checkpoints(config, model, optimizer, scheduler, best=best,
                              extra={"inference": inference_metadata(config, train_dataset)})
    try:
        for epoch in range(first_epoch, int(config["epochs"]) + 1):
            train_losses = _run_epoch(model, train_loader, optimizer, device,
                                      beta, writer, epoch, training=True)
            val_losses = _run_epoch(model, val_loader, optimizer, device,
                                    beta, writer, epoch, training=False)
            scheduler.step()
            losses = {"train": train_losses, "validation": val_losses,
                      "beta": beta}
            for split, values in (("train", train_losses), ("validation", val_losses)):
                record_losses(config, split, values, epoch,
                              logger.log)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            # Validation selects weights independently of test monitoring.
            # Unscheduled checkpoints deliberately carry evaluation=None.
            do_eval = should_evaluate(epoch, int(config["epochs"]), eval_every)
            evaluation = (evaluate(model, val_dataset.series, test_dataset.series,
                                   test_dataset.targets, config, device)
                          if do_eval else None)
            if evaluation is not None:
                _log_evaluation(writer, evaluation, epoch)
                record_test_evaluation(config, evaluation, epoch,
                                       logger.log)
            checkpoints.update(epoch, losses, evaluation, evaluated=do_eval)
        return checkpoints.best
    finally:
        logger.finalize()
