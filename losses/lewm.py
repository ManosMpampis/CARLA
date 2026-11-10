"""LeWM latent prediction, dual SIGReg and optional supervised heads."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.sigreg import SIGReg
from utils.reconstruction_scores import reconstruction_score_map


class LeWMLoss(nn.Module):
    """LeWM loss with optional auxiliary supervision and selectable geometry."""

    def __init__(self, loss2_kind: str = "kl", w_pred: float = 1.0,
                 w_aux: float = 1.0, lambda_sigreg: float = 0.1,
                 lambda_sigreg_tgt: float = 0.1, tau: float = 1.0,
                 sigreg_kwargs: dict | None = None, mask_loss_kind: str = "mse",
                 w_recon: float = 0.0, require_mask: bool = False):
        super().__init__()
        kind = str(loss2_kind).lower()
        if kind not in ("kl", "mse", "l1", "l2"):
            raise ValueError(f"Invalid loss2_kind {loss2_kind}")
        if mask_loss_kind not in ("mse", "bce"):
            raise ValueError("mask_loss_kind must be mse or bce")
        weights = (w_pred, w_aux, lambda_sigreg, lambda_sigreg_tgt, w_recon)
        if any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in weights):
            raise ValueError("loss weights must be finite and nonnegative")
        self.mask_loss_kind = mask_loss_kind
        self.w_recon = float(w_recon)
        self.require_mask = require_mask
        self.loss2_kind = kind
        self.w_pred = float(w_pred)
        self.w_aux = float(w_aux)
        self.lambda_sigreg = float(lambda_sigreg)
        self.lambda_sigreg_tgt = float(lambda_sigreg_tgt)
        self.tau = float(tau)
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be finite and positive")
        self.sigreg = SIGReg(**(sigreg_kwargs or {}))

    def _latent_loss(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if pred.shape != tgt.shape or pred.ndim != 3:
            raise ValueError(f"prediction and target must have the same (B, C, T) shape: {pred.shape}, {tgt.shape}")
        scores, _ = reconstruction_score_map(pred, tgt, self.loss2_kind, tau=self.tau)
        return scores.mean()

    def forward(self, outputs: dict) -> dict:
        pred = outputs["predicted"]["L0"]
        if pred.ndim == 4 and pred.size(1) == 1:
            pred = pred[:, 0]
        zero = pred.new_zeros((), dtype=torch.float32)
        pred_loss = self._latent_loss(pred, outputs["targets"]["L0"]) if self.w_pred else zero
        sig = self.sigreg(outputs["context"]) if self.lambda_sigreg else zero
        sig_tgt = self.sigreg(outputs["targets"]) if self.lambda_sigreg_tgt else zero

        aux_loss = zero
        if self.w_aux:
            logits = outputs.get("mask_logits")
            mask = outputs.get("mask_target")
            if mask is None and isinstance(outputs.get("mask"), dict):
                mask = outputs["mask"].get("input")
            if logits is None or mask is None:
                if self.require_mask:
                    raise ValueError("w_aux requires mask_logits and mask_target")
            else:
                if logits.shape != mask.shape or logits.ndim != 2:
                    raise ValueError("mask_logits and mask_target must have the same (B, T) shape")
                logits, mask = logits.float(), mask.float()
                if self.mask_loss_kind == "bce":
                    aux_loss = F.binary_cross_entropy_with_logits(logits, mask)
                else:
                    aux_loss = F.mse_loss(logits.sigmoid(), mask)

        recon_loss = zero
        if self.w_recon:
            recon, raw_target = outputs.get("recon"), outputs.get("target")
            if (recon is None or raw_target is None or recon.ndim != 3
                    or recon.shape != raw_target.shape):
                raise ValueError("w_recon requires recon and target of the same (B, C, W) shape")
            recon_loss = F.l1_loss(recon.float(), raw_target.float())
        loss = (self.w_pred * pred_loss + self.w_aux * aux_loss + self.w_recon * recon_loss
                + self.lambda_sigreg * sig + self.lambda_sigreg_tgt * sig_tgt)
        result = {"loss": loss, "pred_loss": pred_loss, "aux_loss": aux_loss,
                  "sigreg": sig, "sigreg_tgt": sig_tgt}
        if self.w_recon:
            result["recon_loss"] = recon_loss
        return result
