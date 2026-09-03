import os

import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import AverageMeter


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
                 collator=None, amp: bool = False):
        self.p = p
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.collator = collator
        self.amp = bool(amp) and device.type == "cuda"
        if amp and device.type != "cuda":
            logger.log("amp requested but device is CPU; running fp32")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
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

    def _forward_loss(self, batch):
        ts = self._to_model_input(batch["ts"], self.device)
        mask = None
        if self.collator is not None:
            mask = self.collator(ts.size(0), ts.size(-1), self.model.level_strides)
            mask = {k: v.to(self.device) for k, v in mask.items()}
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
            losses, extras = self._forward_loss(batch)
            if self._codebook_samples is not None and i < 8:
                for name, z in extras["latents"].items():
                    self._codebook_samples[name].append(z.detach().cpu())
            for key, value in losses.items():
                if f"meter_{key}" not in meters:
                    meters[f"meter_{key}"] = AverageMeter(key, ":.4e")
                meters[f"meter_{key}"].update(value.item())
                self.logger.scalar_summary("train", key, value.item(),
                                           epoch * len(loader) + i)

            self.optimizer.zero_grad()
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.model.update_ema()

            if i % 100 == 0:
                var = self.model.latent_variance(extras["latents"])
                self.logger.scalar_summary("train", "latent_var",
                                           var, epoch * len(loader) + i)
                progress = " | ".join(
                    f"{k[6:]} {m.avg:.4e}" for k, m in sorted(meters.items())
                )
                self.logger.log(f"Epoch [{epoch+1}] batch [{i}/{len(loader)}] "
                                f"latent_var {var:.4f} {progress}")

        if self._codebook_samples is not None:
            self.model.codebook.init_from_latents(self._codebook_samples,
                                                  logger=self.logger)
            self._codebook_samples = None
        return {key[6:]: meter.avg for key, meter in meters.items()}

    @torch.no_grad()
    def validate(self, loader) -> float:
        """Latent-prediction validation loss, computed strictly in eval mode."""
        self.model.eval()
        total, count = 0.0, 0
        for batch in loader:
            ts = self._to_model_input(batch["ts"], self.device)
            outputs = self.model(ts)
            losses = self.criterion(outputs)
            total += losses["pred_loss"].item() * ts.size(0)
            count += ts.size(0)
        return total / max(count, 1)

    # ------------------------------------------------------------------ #
    def fit(self, train_loader, val_loader, start_epoch: int = 0,
            best_val_loss: float = np.inf):
        epochs = self.p["epochs"]
        for epoch in range(start_epoch, epochs):
            self.logger.log(f"Epoch {epoch + 1}/{epochs}")
            self.logger.log("-" * 15)
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.log(f"Adjusted learning rate to {lr:.5g}")

            train_losses = self.train_one_epoch(train_loader, epoch)
            self.scheduler.step()

            val_loss = self.validate(val_loader)
            self.logger.scalar_summary("val", "pred_loss", val_loss, epoch + 1)
            self.logger.scalar_summary("", "Learning Rate", lr, epoch + 1)
            self.logger.metrics_summary("Train Loss", train_losses, epoch + 1)
            self.logger.log(f"Epoch [{epoch+1}] val pred_loss {val_loss:.6f} "
                            f"(best {best_val_loss:.6f})")

            if val_loss < best_val_loss or (epoch + 1) == epochs:
                improved = val_loss < best_val_loss
                best_val_loss = min(best_val_loss, val_loss)
                if improved:
                    torch.save(self.model.state_dict(), self.p["jepa_model"])
                self.save_checkpoint(epoch, best_val_loss)
        return best_val_loss

    def save_checkpoint(self, epoch: int, best_val_loss: float) -> None:
        save_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "stage": self.p.get("stage", "pretrain"),
            "anti_collapse": self.model.anti_collapse,
        }
        torch.save(save_dict, self.p["jepa_checkpoint"])

    @staticmethod
    def _migrate_legacy_keys(state: dict) -> dict:
        """Remap pre-rebuild encoder submodule names (shared helper)."""
        from models.encoder import remap_legacy_encoder_keys

        return remap_legacy_encoder_keys(state)

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
        model.load_state_dict(Trainer._migrate_legacy_keys(checkpoint["model"]))
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
        model.load_state_dict(Trainer._migrate_legacy_keys(state), strict=strict)
        if logger is not None:
            logger.log(f"Loaded weights from {path}")
