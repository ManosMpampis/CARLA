"""AE/VAE training and shared three-mode anomaly evaluation."""

import os
import random

import numpy as np
import torch

from data.jepa_dataset import JEPADataset
from models.ae_baseline import reconstruction_objective
from models.builders import get_ae_model, get_vae_model
from utils.common_config import (get_jepa_datasets, get_optimizer, get_scheduler,
                                 get_train_dataloader, get_val_dataloader)
from utils.config import create_config
from utils.evaluation_schedule import evaluation_interval, should_evaluate
from utils.checkpointing import Checkpoints, inference_metadata
from utils.utils import Logger
from utils.evaluation_logging import (log_evaluation, loss_metrics,
                                      log_losses, record_losses, latent_variance, truncate_epoch_history)
from utils.evaluator import record_evaluation, evaluate


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

def _run_epoch(model, loader, optimizer, device, beta: float, writer,
               epoch: int, training: bool) -> dict:
    model.train(training)
    totals = {}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            x = batch["ts"].transpose(1, 2).contiguous().to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(x, sample=training and model.variational)
            losses = reconstruction_objective(outputs, x, beta)
            if training:
                losses["loss"].backward()
                optimizer.step()
            latent = outputs["latent"].detach()
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

def run_arm(arm, args, update_dictionary=None):
    """Single AE/VAE entry: train normally, or score the same saved experiment."""
    if str(stage).lower() == "score":
        config = create_config(args.config_env, args.config_exp, args.fname,
                               args.version, update_dictionary=update_dictionary)
        return score_arm(arm, args, config)
    return train_arm(arm, args, update_dictionary)


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
    last_path = config["jepa_checkpoint"]
    if not os.path.exists(last_path):
        last_path = os.path.join(out_dir, "last.pth.tar")  # legacy AE/VAE resume
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
    eval_every = evaluation_interval(config)
    test_dataset = (JEPADataset(config, train=False)
                    if eval_every > 0 else None)
    train_loader = get_train_dataloader(config, train_dataset)
    val_loader = get_val_dataloader(config, val_dataset)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    logger = Logger(config["version"], verbose=3, file_path=out_dir, delete_files=False)
    beta = float(config.get("beta", 0.3)) if arm == "vae" else 0.0
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    first_epoch, best = 1, None
    if os.path.exists(last_path):
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best = checkpoint.get("best")
        first_epoch = int(checkpoint["epoch"]) + 1
        initialization_source = checkpoint.get("initialize_from")
    config["initialize_from"] = initialization_source
    if first_epoch > 1:
        logger.resume_from(first_epoch - 1)
        truncate_epoch_history(out_dir, first_epoch - 1)
    writer = logger.experiment
    checkpoints = Checkpoints(config, model, optimizer, scheduler, best=best,
                              extra={"inference": inference_metadata(config, train_dataset)})
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
        if config.get("benchmark_protocol") == "tsb_ad":
            logger.log(f"TSB-AD protocol: test monitoring every {eval_every} epochs; "
                       "default checkpoint selected by validation loss")
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
            evaluation = (evaluate(model, train_dataset.full_train_series, test_dataset.series,
                                   test_dataset.targets, config, device)
                          if do_eval else None)
            if evaluation is not None:
                log_evaluation(writer, evaluation, epoch)
                record_evaluation(config, evaluation, epoch,
                                       logger.log)
            checkpoints.update(epoch, losses, evaluation, evaluated=do_eval)
        return checkpoints.best
    finally:
        logger.finalize()
