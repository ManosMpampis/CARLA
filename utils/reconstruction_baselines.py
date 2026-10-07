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
from models.ae_baseline import ReconstructionBaseline, reconstruction_objective
from utils.common_config import (get_jepa_datasets, get_optimizer, get_scheduler,
                                 get_train_dataloader, get_val_dataloader)
from utils.config import create_config
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


def _evaluate_protocol(scores, labels, clean_scores, quantile, vus_window):
    calibrated = float(np.quantile(clean_scores, quantile))
    oracle = _oracle_threshold(scores, labels)
    vus = get_range_vus_roc(scores, labels, vus_window)
    return {
        "calibrated": _classification(scores, labels, calibrated),
        "oracle": _classification(scores, labels, oracle),
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


def evaluate(model, val_series, test_series, test_labels, config, device) -> dict:
    """Distinct window and overlap-averaged time-series evaluation."""
    options = evaluation_options(config)
    window, stride = options.pop("wsz"), options.pop("stride")
    batch_size = int(config.get("score_batch_size", config["batch_size"]))
    clean = score_both(model, val_series, window, stride, batch_size, device,
                       **options)
    test = score_both(model, test_series, window, stride, batch_size, device,
                      **options)
    return evaluate_from_scores(clean, test, test_labels, config)


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
          config, train_dataset, selection) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "losses": losses,
        "evaluation": evaluation,
        "selection": selection,
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


def train_arm(arm: str, args) -> dict:
    if arm not in {"ae", "vae"}:
        raise ValueError(f"unknown reconstruction arm {arm}")
    config = create_config(args.config_env, args.config_exp, args.fname,
                           args.version)
    if config.get("arm") != arm:
        raise ValueError(f"config arm {config.get('arm')} does not match {arm}")
    _seed(int(config.get("seed", 4)))
    device = _device(config)
    model = ReconstructionBaseline(dict(config["model_kwargs"]),
                                   dict(config.get("recon_kwargs", {})),
                                   variational=arm == "vae",
                                   score_mode=config.get("score_mode", "l1")).to(device)
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
    writer = SummaryWriter(os.path.join(out_dir, "tensorboard"))
    beta = float(config.get("beta", 0.3)) if arm == "vae" else 0.0
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    eval_every = int(config.get("eval_every", 1))
    if eval_every < 1:
        raise ValueError("eval_every must be positive")
    best = {"validation_loss": float("inf")}
    for protocol in ("window", "timeseries"):
        for key in ("calibrated_f1", "oracle_f1", "vus_pr", "vus_roc"):
            best[f"{protocol}_{key}"] = float("-inf")
    last_path = os.path.join(out_dir, "last.pth.tar")
    first_epoch = 1
    if os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best.update(checkpoint["best"])
        first_epoch = int(checkpoint["epoch"]) + 1
    try:
        for epoch in range(first_epoch, int(config["epochs"]) + 1):
            train_losses = _run_epoch(model, train_loader, optimizer, device,
                                      beta, writer, epoch, training=True)
            val_losses = _run_epoch(model, val_loader, optimizer, device,
                                    beta, writer, epoch, training=False)
            scheduler.step()
            losses = {"train": train_losses, "validation": val_losses,
                      "beta": beta}
            # A newly selected validation checkpoint must also contain that
            # epoch's evaluation values, even between scheduled intervals.
            do_eval = (epoch % eval_every == 0 or
                       epoch == int(config["epochs"]) or
                       val_losses["loss"] < best["validation_loss"])
            evaluation = (evaluate(model, val_dataset.series, test_dataset.series,
                                   test_dataset.targets, config, device)
                          if do_eval else None)
            if evaluation is not None:
                _log_evaluation(writer, evaluation, epoch)
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
                          train_dataset, selection)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch,
                        "best": best, "losses": losses, "evaluation": evaluation},
                       last_path)
            with open(os.path.join(out_dir, "epochs.jsonl"), "a") as stream:
                stream.write(json.dumps({"epoch": epoch, "losses": losses,
                                         "evaluation": evaluation}) + "\n")
            print(f"{arm.upper()} epoch {epoch}: val loss={val_losses['loss']:.6f}"
                  + ("; test evaluated" if do_eval else ""), flush=True)
        with open(os.path.join(out_dir, "best_metrics.json"), "w") as stream:
            json.dump(best, stream, indent=2)
        return best
    finally:
        writer.close()
