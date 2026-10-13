"""Independent AE/VAE training and two-granularity reconstruction evaluation."""

import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

from data.jepa_dataset import JEPADataset
from metrics.affiliation.generics import convert_vector_to_events
from metrics.f1_score_f1_pa import event_f1
from metrics.vus.metrics import get_range_vus_roc
from models.ae_baseline import reconstruction_objective
from models.builders import get_ae_model, get_vae_model
from utils.common_config import (get_jepa_datasets, get_optimizer, get_scheduler,
                                 get_train_dataloader, get_val_dataloader)
from utils.config import create_config, entry_overrides, load_experiment_config
from utils.evaluation_schedule import evaluation_interval, should_evaluate
from utils.scoring import (aggregate_score_maps, covered_evaluation_view,
                           evaluation_options)
from utils.utils import Logger


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
    if not np.any(labels):
        raise ValueError("test labels contain no anomalies; oracle F1 undefined")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(2 * precision[:-1] * recall[:-1],
                   precision[:-1] + recall[:-1],
                   out=np.zeros_like(precision[:-1]),
                   where=(precision[:-1] + recall[:-1]) > 0)
    # Predictions use strict >. Move one representable float below the
    # selected observed score so that the maximizing score remains included.
    return float(np.nextafter(thresholds[int(np.argmax(f1))], -np.inf))


def _classification(scores, labels, threshold) -> dict:
    pred = (scores > threshold).astype(np.int64)
    return {"threshold": float(threshold),
            "f1_no_pa": float(f1_score(labels, pred, zero_division=0)),
            "event_f1_no_pa": float(event_f1(
                labels, convert_vector_to_events(labels), pred))}


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
    vus = get_range_vus_roc(scores, labels, vus_window)
    calibrated_metrics = _classification(scores, labels, calibrated)
    calibrated_metrics["calibration_quantile"] = float(quantile)
    oracle_metrics = _classification(scores, labels, oracle)
    oracle_metrics.update(calibration_at_threshold(clean_scores, oracle))
    return {
        "calibrated": calibrated_metrics,
        "oracle": oracle_metrics,
        "vus_pr": float(vus["VUS_PR"]),
        "vus_roc": float(vus["VUS_ROC"]),
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
    totals = {"loss": 0.0, "mse": 0.0, "kl": 0.0}
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
            for name, value in losses.items():
                totals[name] += float(value.detach()) * x.size(0)
            count += x.size(0)
    if count == 0:
        raise ValueError("empty data loader; lower batch size or window length")
    averages = {name: value / count for name, value in totals.items()}
    prefix = "train" if training else "validation"
    for name, value in averages.items():
        writer.add_scalar(f"{prefix}/{name}", value, epoch)
    return averages


def _save(path, model, optimizer, scheduler, epoch, losses, evaluation,
          config, train_dataset, selection, initialize_from=None) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "losses": losses,
        "evaluation": evaluation,
        "selection": selection,
        "initialize_from": initialize_from,
        "inference": {
            "arm": "vae" if model.variational else "ae",
            "model_kwargs": dict(config["model_kwargs"]),
            "recon_kwargs": dict(config.get("recon_kwargs", {})),
            "window_length": int(config["wsz"]),
            "stride": int(config["stride"]),
            "score_mode": model.score_mode,
            "score": "mean_timestep_score_map",
            "overlap": "cover_count_mean",
            "vae_latent": "posterior_mean" if model.variational else None,
            "normalization_mean": np.asarray(train_dataset.mean).tolist(),
            "normalization_std": np.asarray(train_dataset.std).tolist(),
        },
        "config": dict(config),
    }
    torch.save(payload, path)


def _log_evaluation(writer, evaluation, epoch):
    for protocol, metrics in evaluation.items():
        for threshold_source in ("calibrated", "oracle"):
            for name, value in metrics[threshold_source].items():
                writer.add_scalar(f"test/{protocol}/{threshold_source}/{name}",
                                  value, epoch)
        writer.add_scalar(f"test/{protocol}/vus_pr", metrics["vus_pr"], epoch)
        writer.add_scalar(f"test/{protocol}/vus_roc", metrics["vus_roc"], epoch)


def record_test_evaluation(config, evaluation, epoch, log):
    """Persist every scheduled evaluation and display all available metrics."""
    with open(os.path.join(config["jepa_dir"], "test_evaluations.jsonl"), "a") as stream:
        stream.write(json.dumps({"epoch": epoch, "evaluation": evaluation}) + "\n")
    for protocol, metrics in evaluation.items():
        parts = []
        for source in ("calibrated", "oracle"):
            values = ", ".join(f"{name}={value:.6f}"
                               for name, value in metrics[source].items())
            parts.append(f"{source}: {values}")
        parts.extend(f"{name}={metrics[name]:.6f}" for name in ("vus_pr", "vus_roc"))
        log(f"Test eval [{epoch}] {protocol} | " + " | ".join(parts))


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
    return metrics["timeseries"]["calibrated"]["f1_no_pa"]


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
    out_dir = config["jepa_dir"]
    last_path = os.path.join(out_dir, "last.pth.tar")
    initialization_source = None
    if config.get("initialize_from") and not os.path.exists(last_path):
        # Fine-tuning starts a new optimizer and schedule. An existing run's
        # complete state always takes precedence over a weights-only source.
        from utils.trainer import Trainer

        initialization_source = str(config["initialize_from"])
        Trainer.load_weights(initialization_source, model, strict=True)
    train_dataset, val_dataset = get_jepa_datasets(config)
    if not isinstance(train_dataset, JEPADataset) or train_dataset._corpus is not None:
        raise ValueError("these comparison arms require a single machine")
    benchmark = config.get("benchmark_protocol") == "tsb_ad"
    eval_every = 0 if benchmark else evaluation_interval(config)
    trace_enabled = not benchmark and config.get("decision_trace") is not False
    test_dataset = (JEPADataset(config, train=False)
                    if eval_every > 0 or trace_enabled else None)
    train_loader = get_train_dataloader(config, train_dataset)
    val_loader = get_val_dataloader(config, val_dataset)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    beta = float(config.get("beta", 0.3)) if arm == "vae" else 0.0
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    best = {"validation_loss": float("inf")}
    for protocol in ("window", "timeseries", "non_overlap_window"):
        for key in ("calibrated_f1", "oracle_f1", "vus_pr", "vus_roc"):
            best[f"{protocol}_{key}"] = float("-inf")
    first_epoch = 1
    if os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best.update(checkpoint["best"])
        first_epoch = int(checkpoint["epoch"]) + 1
        initialization_source = checkpoint.get("initialize_from")
    logger = Logger(config.get("version", "default"), verbose=2, file_path=out_dir,
                    use_tensorboard=False, delete_files=False)
    writer = SummaryWriter(os.path.join(out_dir, "tensorboard"))
    try:
        logger.log(f"{arm.upper()} training; train/validation batches "
                   f"{len(train_loader)}/{len(val_loader)}")
        logger.log_hyperparams(config)
        if initialization_source:
            logger.log(f"Weights initialized from {initialization_source}")
        if os.path.exists(last_path):
            logger.log(f"Restart from checkpoint {last_path}; next epoch {first_epoch}")
            from utils.trainer import Trainer

            Trainer.apply_resume_learning_rate(config, optimizer, scheduler, logger)
        if benchmark:
            logger.log("TSB-AD protocol: training and selection use train data only")
        for epoch in range(first_epoch, int(config["epochs"]) + 1):
            train_losses = _run_epoch(model, train_loader, optimizer, device,
                                      beta, writer, epoch, training=True)
            val_losses = _run_epoch(model, val_loader, optimizer, device,
                                    beta, writer, epoch, training=False)
            scheduler.step()
            losses = {"train": train_losses, "validation": val_losses,
                      "beta": beta}
            # Validation selects weights independently of test monitoring.
            # Unscheduled checkpoints deliberately carry evaluation=None.
            do_eval = should_evaluate(epoch, int(config["epochs"]), eval_every)
            evaluation = (evaluate(model, val_dataset.series, test_dataset.series,
                                   test_dataset.targets, config, device)
                          if do_eval else None)
            if evaluation is not None:
                _log_evaluation(writer, evaluation, epoch)
                record_test_evaluation(config, evaluation, epoch,
                                       lambda message: print(message, flush=True))
            candidates = {"validation_loss": (val_losses["loss"], "min",
                                              "best_validation_loss.pth.tar",
                                              {"metric": "loss", "source": "validation"})}
            if evaluation is not None:
                for protocol, metrics in evaluation.items():
                    for source in ("calibrated", "oracle"):
                        key = f"{protocol}_{source}_f1"
                        candidates[key] = (
                            metrics[source]["f1_no_pa"], "max",
                            f"best_test_{key}.pth.tar",
                            {"metric": "f1_no_pa", "source": "test",
                             "procedure": protocol, "threshold_source": source,
                             "threshold": metrics[source]["threshold"]})
                    for metric in ("vus_pr", "vus_roc"):
                        key = f"{protocol}_{metric}"
                        candidates[key] = (
                            metrics[metric], "max", f"best_test_{key}.pth.tar",
                            {"metric": metric, "source": "test",
                             "procedure": protocol,
                             "threshold_source": "independent"})
            for key, (value, mode, filename, selection) in candidates.items():
                improved = value < best[key] if mode == "min" else value > best[key]
                if improved and np.isfinite(value):
                    best[key] = float(value)
                    _save(os.path.join(out_dir, filename), model, optimizer,
                          scheduler, epoch, losses, evaluation, config,
                          train_dataset, selection, initialization_source)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch,
                        "best": best, "losses": losses, "evaluation": evaluation,
                        "initialize_from": initialization_source,
                        "benchmark_protocol": config.get("benchmark_protocol")},
                       last_path)
            with open(os.path.join(out_dir, "epochs.jsonl"), "a") as stream:
                stream.write(json.dumps({"epoch": epoch, "losses": losses,
                                         "evaluation": evaluation}) + "\n")
            message = (f"{arm.upper()} epoch {epoch}: val loss={val_losses['loss']:.6f}"
                       + ("; test evaluated" if do_eval else ""))
            print(message, flush=True)
            logger.log(message)
        with open(os.path.join(out_dir, "best_metrics.json"), "w") as stream:
            json.dump(best, stream, indent=2)
        selected_weights = os.path.join(out_dir, "best_validation_loss.pth.tar")
        if trace_enabled and os.path.isfile(selected_weights):
            # Ordinary AE/VAE training also exports the final validation-selected
            # replay; --score and suite scoring refresh it through score_experiment.
            from utils.decision_trace import write_final_decision_trace

            selected = torch.load(selected_weights, map_location="cpu", weights_only=False)
            model.load_state_dict(selected["model"])
            options = evaluation_options(config)
            window, stride = options.pop("wsz"), options.pop("stride")
            clean = score_both(model, val_dataset.series, window, stride,
                               int(config.get("score_batch_size", config["batch_size"])),
                               device, **options)
            quantile = float(config.get("calibration_kwargs", {}).get(
                "quantile", config.get("calibration_quantile", 0.99)))
            threshold = float(np.quantile(
                clean["timeseries_scores"][clean["cover_counts"] > 0], quantile))
            artifact = write_final_decision_trace(
                model, device, test_dataset.series, config, threshold,
                options={"wsz": window, "stride": stride, **options},
                threshold_operator="gt", labels=test_dataset.targets)
            print(f"{arm.upper()} validation-selected decision replay: {artifact['html']}", flush=True)
        return best
    finally:
        writer.close()
        logger.finalize()
