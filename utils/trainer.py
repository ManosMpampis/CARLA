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
        self._codebook_samples: dict[str, list] | None = None

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

    def _make_mask(self, collator, batch_size: int, window: int,
                     level_strides, ts=None):
        if collator is None:
            return None
        try:
            mask = collator(batch_size, window, list(level_strides), ts)
        except TypeError:
            mask = collator(batch_size, window, list(level_strides))
        return {k: v.to(self.device) for k, v in mask.items()}

    def _forward_loss(self, batch):
        ts = self._to_model_input(batch["ts"], self.device)
        mask = self._make_mask(self.collator, ts.size(0), ts.size(-1),
                               self.model.level_strides, ts)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.amp):
            outputs = self.model(ts, mask=mask)
            losses = self.criterion(outputs)
        return losses, {"latents": outputs["latents"], "ts": ts}

    def train_one_epoch(self, loader, epoch: int) -> dict:
        self.model.train()
        if getattr(self.model, "encoder_frozen", False):
            # frozen adaptation: BN stats of the encoder must not move
            self.model.encoder.eval()
        if self.model.target_encoder is not None:
            self.model.target_encoder.train_mode()
        if self.model.codebook is not None and epoch == 0 \
                and not self.model.codebook.initialized:
            # collect first-epoch latents for the k-means warmup
            samples: dict[str, list] = {name: [] for name in self.model.level_names}
            self._codebook_samples = samples
        meters = {}
        for i, batch in enumerate(loader):
            self.optimizer.zero_grad()

            losses, extras = self._forward_loss(batch)
            if self._codebook_samples is not None and i < 8:
                for name, z in extras["latents"].items():
                    self._codebook_samples[name].append(z.detach().cpu())
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.model.update_ema()
            self.model.update_running_stats(extras["latents"])

            values = loss_metrics(losses)
            values["latent_var"] = latent_variance(extras["latents"])
            for key, value in values.items():
                if key not in meters:
                    meters[key] = AverageMeter(key, ":.4e")
                scalar = float(value.detach()) if torch.is_tensor(value) else float(value)
                meters[key].update(scalar, extras["ts"].size(0))

        if self._codebook_samples is not None:
            self.model.codebook.init_from_latents(self._codebook_samples,
                                                  logger=self.logger)
            self._codebook_samples = None
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
            mask = self._make_mask(self.val_collator, ts.size(0), ts.size(-1),
                                   self.model.level_strides, ts)
            outputs = self.model(ts, mask=mask)
            losses = self.criterion(outputs)
            values = loss_metrics(losses)
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
            do_eval = eval_fn is not None and should_evaluate(step, epochs, eval_every)
            evaluation = eval_fn(step) if do_eval else None
            checkpoints.update(step, {"train": train_losses, "validation": val_losses},
                               evaluation, evaluated=do_eval, scaler=self.scaler)
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
            return 0, np.inf
        logger.log(f"Restart from checkpoint {path}")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("next_epoch", checkpoint["epoch"] + 1)
        best = checkpoint.get("best_val_loss", np.inf)
        return start_epoch, best

    @staticmethod
    def load_weights(path, model, logger=None, strict: bool = True):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint \
            else checkpoint
        model.load_state_dict(state, strict=strict)
        if logger is not None:
            logger.log(f"Loaded weights from {path}")
