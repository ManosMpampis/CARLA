import os

import numpy as np
import torch

from utils.utils import AverageMeter
from utils.evaluation_schedule import should_evaluate
from utils.checkpointing import Checkpoints, inference_metadata
from utils.evaluation_logging import (loss_metrics, record_losses, latent_variance,
                                      truncate_epoch_history)


class Trainer:
    """Stage runner for JEPA pretraining/adaptation.

    - Mixed precision behind the ``amp`` config flag: bf16 autocast +
      GradScaler.
    - Checkpoints follow the repo's existing resume format
      (model/optimizer/scheduler/epoch/next_epoch + best-metric fields),
      extended minimally with stage metadata.
    - Validation statistics are computed in eval mode: BatchNorm buffers
      must not inflate the selection signal (LeWorldModel-repro lesson).
    """

    def __init__(self, p, model, criterion, optimizer, scheduler, device, logger, amp: bool = False):
        self.p = p
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.quantile = ...
        self.amp = bool(amp)
        self.scaler = torch.amp.GradScaler(device, enabled=self.amp)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_model_input(batch_ts: torch.Tensor, device) -> torch.Tensor:
        # Dataset windows are (W, C); the model consumes (B, C, W).
        ts = batch_ts.to(device, non_blocking=True)
        if ts.ndim == 2:
            ts = ts.unsqueeze(1)
        elif ts.ndim == 3:
            ts = ts.transpose(1, 2)
        return ts.contiguous()

    def _forward_loss(self, batch):
        # TODO: check how the input engine will be used, maybe run self.model.train
        ts = self._to_model_input(batch["ts"], self.device)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.amp):
            outputs = self.model(ts)
            losses = self.criterion(outputs)
        return losses, {"latents": outputs["latents"], "ts": ts}

    def train_one_epoch(self, loader, epoch: int) -> dict:
        self.model.train()
        if getattr(self.model, "encoder_frozen", False):
            self.model.encoder.eval()
        meters = {}
        for i, batch in enumerate(loader):
            self.optimizer.zero_grad()

            losses, extras = self._forward_loss(batch)
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            values = loss_metrics(losses)
            values["latent_var"] = latent_variance(extras["latents"])
            for key, value in values.items():
                if key not in meters:
                    meters[key] = AverageMeter(key, ":.4e")
                scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
                meters[key].update(scalar, extras["ts"].size(0))

        return {key: meter.avg for key, meter in meters.items()}

    @torch.no_grad()
    def validate(self, loader) -> float:
        """Validation loss in eval mode; masked when val_collator is set.

        Masked-prediction arms must select on the masked task (the scoring
        task), not on dense self-reconstruction: dense val loss can collapse
        to ~0 while probe errors on unseen windows blow up, destroying the
        train-calibrated threshold scale. Default None = legacy dense path.
        """
        self.model.eval()
        totals, count = {}, 0
        for batch in loader:
            ts = self._to_model_input(batch["ts"], self.device)
            losses, extras = self._forward_loss(batch)
            
            values = loss_metrics(losses)
            values["latent_var"] = latent_variance(extras["latents"])
            for key, value in values.items():
                scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
                totals[key] = totals.get(key, 0.0) + scalar * ts.size(0)
            count += ts.size(0)
        if count == 0:
            raise ValueError("empty validation loader")
        self.validation_metrics = {key: value / count for key, value in totals.items()}
        return self.validation_metrics["loss"]

    # ------------------------------------------------------------------ #
    def fit(self, train_loader, val_loader, start_epoch: int = 0,
            best_val_loss: float = np.inf,
            eval_every: int = 0, eval_fn=None):
        """Log complete losses and save validation/oracle/periodic snapshots."""
        if self.p.get("benchmark_protocol") == "tsb_ad":
            self.logger.log_hyperparams(self.p)
        if start_epoch:
            self.logger.resume_from(start_epoch)
            truncate_epoch_history(self.p["jepa_dir"], start_epoch)
        previous = {}
        path = self.p.get("jepa_checkpoint")
        if start_epoch and path and os.path.exists(path):
            previous = torch.load(path, map_location="cpu", weights_only=False)
            if "scaler" in previous:
                self.scaler.load_state_dict(previous["scaler"])
        best = {**previous.get("best", {}), "validation_loss": best_val_loss}
        checkpoints = Checkpoints(self.p, self.model, self.optimizer, self.scheduler,
                                  best=best, extra={
                                      "anti_collapse": self.model.anti_collapse,
                                      "inference": inference_metadata(
                                          self.p, getattr(train_loader, "dataset", None))})
        epochs = int(self.p["epochs"])
        for epoch in range(start_epoch, epochs):
            step = epoch + 1
            train_losses = self.train_one_epoch(train_loader, epoch)
            self.scheduler.step()
            self.validate(val_loader)
            val_losses = self.validation_metrics
            for split, values in (("train", train_losses), ("validation", val_losses)):
                self.logger.metrics_summary(split, values, step)
                record_losses(self.p, split, values, step, self.logger.log)
            self.logger.scalar_summary("train", "learning_rate",
                                       self.optimizer.param_groups[0]["lr"], step)

            if should_evaluate(step, epochs, eval_every):
                self.Evaluator.calibrate(self.model, train_loader.dataset, self.quantile) #TODO: check if I need to calibrate with validation sub_set
                evaluation_metrics = self.Evaluator(self.model, test_loader.dataset)
                for phase, metric in evaluation_metrics.items():
                    self.logger.metrics_summary(phase, metric, step)
            
            checkpoints.update(step, {"train": train_losses, "validation": val_losses},
                               evaluation, evaluated=should_evaluate(step, epochs, eval_every), scaler=self.scaler)
        return checkpoints.best["validation_loss"]

    @staticmethod
    def resume(p, model, optimizer, scheduler, logger, map_location="cpu"):
        """Load run state from an existing checkpoint, if any.

        Returns (start_epoch, best_val_loss). Old-format checkpoints remain
        loadable: missing metadata keys fall back to sane defaults.
        """
        path = p["jepa_checkpoint"]
        if not os.path.exists(path):
            logger.log(f"No checkpoint file at {path}")
            if p.get("initialize_from"):
                Trainer.initialize_weights(p["initialize_from"], model, logger)
            return 0, np.inf
        logger.log(f"Restart from checkpoint {path}")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if checkpoint.get("initialize_from"):
            p["initialize_from"] = checkpoint["initialize_from"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        Trainer.apply_resume_learning_rate(p, optimizer, scheduler, logger)
        start_epoch = checkpoint.get("next_epoch", checkpoint["epoch"] + 1)
        best = checkpoint.get("best_val_loss", np.inf)
        return start_epoch, best

    @staticmethod
    def apply_resume_learning_rate(p, optimizer, scheduler, logger=None):
        """Apply an explicit tuning rate after restoring optimizer/scheduler state."""
        if p.get("resume_learning_rate") is None:
            return
        lr = float(p["resume_learning_rate"])
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("resume_learning_rate must be finite and positive")
        rates = [lr] * len(optimizer.param_groups)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] = lr

        def update_schedule(current):
            for attr in ("base_lrs", "_last_lr"):
                if hasattr(current, attr):
                    setattr(current, attr, list(rates))
            for child in getattr(current, "_schedulers", []):
                update_schedule(child)

        update_schedule(scheduler)
        if logger is not None:
            logger.log(f"Resumed optimizer learning rate overridden to {lr}")

    @staticmethod
    def initialize_weights(path, model, logger=None):
        """Strict model warm start, allowing only an encoder freezing change."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        state = dict(state)
        if "_extra_state" in state and hasattr(model, "_task"):
            # Freezing controls optimization, not the prediction task. Keep
            # every architecture/crop/query check used by strict resume.
            task = dict(state["_extra_state"])
            if "freeze_encoder" in task:
                task["freeze_encoder"] = model.get_extra_state()["freeze_encoder"]
                state["_extra_state"] = task
        model.load_state_dict(state, strict=True)
        if logger is not None:
            logger.log(f"Initialized model weights from {path}; optimizer starts fresh")

    @staticmethod
    def load_weights(path, model, logger=None, strict: bool = True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint \
            else checkpoint
        model.load_state_dict(state, strict=strict)
        if logger is not None:
            logger.log(f"Loaded weights from {path}")
