import os

import numpy as np
import torch

from utils.utils import AverageMeter
from utils.evaluation_schedule import should_evaluate
from utils.checkpointing import Checkpoints, inference_metadata
from utils.evaluation_logging import (record_losses, latent_variance,
                                      truncate_epoch_history)
from utils.randomness import isolated_rng, restore_rng_state, isolated_module_rng, restore_module_rng


class Trainer:
    """Stage runner for JEPA pretraining/adaptation.

    - Mixed precision behind the ``amp`` config flag: bf16 autocast +
      GradScaler on CUDA; silently fp32 elsewhere (scoring is always fp32).
    - Checkpoints follow the repo's existing resume format
      (model/optimizer/scheduler/epoch/next_epoch + best-metric fields),
      extended minimally with stage metadata.
    - Validation statistics are computed in eval mode: BatchNorm buffers
      must not inflate the selection signal (LeWorldModel-repro lesson).
    """

    def __init__(self, p, model, criterion, optimizer, scheduler, device, logger,
                 collator=None, amp: bool = False, val_collator=None):
        self.p = p
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.collator = collator
        self.amp = bool(amp)
        self.val_collator = val_collator
        self.scaler = torch.amp.GradScaler(device, enabled=self.amp)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_model_input(batch_ts: torch.Tensor, device) -> torch.Tensor:
        # Dataset windows are (W, C); the model consumes (B, C, W).
        ts = batch_ts.float().to(device, non_blocking=True)
        if ts.ndim == 2:
            ts = ts.unsqueeze(1)
        elif ts.ndim == 3:
            ts = ts.transpose(1, 2)
        return ts.contiguous()

    def _make_mask(self, collator, ts):
        if collator is None:
            return None
        batch_size, window = ts.size(0), ts.size(-1)
        level_strides = self.model.level_strides
        try:
            mask = collator(batch_size, window, list(level_strides), ts)
        except TypeError:
            mask = collator(batch_size, window, list(level_strides))
        return {k: v.to(self.device) for k, v in mask.items()}

    def _forward_loss(self, batch):
        ts = self._to_model_input(batch["ts"], self.device)
        mask = self._make_mask(self.collator, ts)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.amp):
            action = batch.get("action")
            if action is None:
                outputs = self.model(ts, mask=mask)
            else:
                outputs = self.model(ts, mask=mask, action=action.to(self.device))
            losses = self.criterion(outputs)
        return losses, {"latents": outputs["latents"], "ts": ts}

    def train_one_epoch(self, loader, epoch: int) -> dict:
        self.model.train()
        if getattr(self.model, "encoder_frozen", False):
            # frozen adaptation: BN stats of the encoder must not move
            self.model.encoder.eval()
        target_encoder = getattr(self.model, "target_encoder", None)
        if target_encoder is not None:
            target_encoder.train_mode()
        meters = {}
        update_ema = getattr(self.model, "update_ema", None)
        update_stats = getattr(self.model, "update_running_stats", None)
        for i, batch in enumerate(loader):
            self.optimizer.zero_grad()

            losses, extras = self._forward_loss(batch)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("training objective became nonfinite")
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if update_ema is not None:
                update_ema()
            if update_stats is not None:
                update_stats(extras["latents"])

            values = dict(losses)
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
        with isolated_rng(self.p.get("validation_seed", int(self.p.get("seed", 4)) + 100003)), isolated_module_rng(self.criterion):
            return self._validate(loader)

    def _validate(self, loader):
        self.model.eval()
        totals, count = {}, 0
        for batch in loader:
            ts = self._to_model_input(batch["ts"], self.device)
            mask = self._make_mask(self.val_collator, ts)
            action = batch.get("action")
            if action is None:
                outputs = self.model(ts, mask=mask)
            else:
                outputs = self.model(ts, mask=mask, action=action.to(self.device))
            losses = self.criterion(outputs)
            values = dict(losses)
            values["latent_var"] = latent_variance(outputs["latents"])
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
            restore_module_rng(self.criterion, previous.get("criterion_rng_state"))
        primary_split = "validation" if self.p.get("validation_available", True) else "train"
        best = {**previous.get("best", {}), f"{primary_split}_loss": best_val_loss}
        checkpoints = Checkpoints(self.p, self.model, self.optimizer, self.scheduler,
                                  best=best, criterion=self.criterion, extra={
                                      "anti_collapse": getattr(self.model, "anti_collapse", "none"),
                                      "inference": inference_metadata(
                                          self.p, getattr(train_loader, "dataset", None))})
        epochs = int(self.p["epochs"])
        for epoch in range(start_epoch, epochs):
            step = epoch + 1
            train_losses = self.train_one_epoch(train_loader, epoch)
            self.scheduler.step()
            if self.p.get("validation_available", True):
                self.validate(val_loader)
                val_losses = self.validation_metrics
            else:
                val_losses = None
            for split, values in (("train", train_losses), ("validation", val_losses)):
                if values is None:
                    continue
                self.logger.metrics_summary(split, values, step)
                record_losses(self.p, split, values, step, self.logger.log)
            self.logger.scalar_summary("train", "learning_rate",
                                       self.optimizer.param_groups[0]["lr"], step)
            do_eval = eval_fn is not None and should_evaluate(step, epochs, eval_every)
            evaluation = eval_fn(step) if do_eval else None
            checkpoints.update(step, {"train": train_losses, "validation": val_losses},
                               evaluation, evaluated=do_eval, scaler=self.scaler)
        return checkpoints.best[f"{primary_split}_loss"]

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
        Trainer.validate_run_contract(p, checkpoint)
        if checkpoint.get("initialize_from"):
            p["initialize_from"] = checkpoint["initialize_from"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        Trainer.apply_resume_learning_rate(p, optimizer, scheduler, logger)
        start_epoch = checkpoint.get("next_epoch", checkpoint["epoch"] + 1)
        best = checkpoint.get("best_primary_loss", checkpoint.get("best_val_loss", np.inf))
        restore_rng_state(checkpoint.get("rng_state"))
        return start_epoch, best

    @staticmethod
    def validate_run_contract(config, checkpoint):
        for key, field in (("run_contract", "_run_contract"), ("preprocessing", "_preprocessing")):
            saved, current = checkpoint.get(key), config.get(field)
            if saved is not None and current is not None and saved != current:
                raise ValueError(f"checkpoint {key} differs from configured run")

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
    def initialize_backbone(model, source, prefix="encoder."):
        """Transfer only the source encoder to the destination backbone."""
        if model.backbone is None:
            raise ValueError("pretrained_from requires a backbone module")
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        if prefix:
            state = {key.removeprefix(prefix): value for key, value in state.items()
                     if key.startswith(prefix)}
        if not state:
            raise ValueError(f"no backbone weights with prefix {prefix!r} in {source}")
        model.backbone.load_state_dict(state, strict=True)

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
