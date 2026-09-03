"""Backward-compatible JEPA criterion on top of the clean pieces.

Keeps the `criterion: jepa` config key working: dense L1 part prediction
plus isotropy regularization (on disposable projections when present)
plus optional codebook term. New per-head losses live beside it.
"""
import torch.nn as nn

from losses.prediction import DensePartLoss
from losses.sigreg import SIGReg


class JEPALoss(nn.Module):
    """Dense latent-prediction criterion (stable external contract)."""

    def __init__(self, lambda_sigreg: float = 0.0, sigreg_kwargs: dict | None = None,
                 target_norm: str | None = None, level_weights: dict | None = None,
                 lambda_codebook: float = 1.0):
        super().__init__()
        self.pred = DensePartLoss(level_weights=level_weights, target_norm=target_norm)
        self.lambda_sigreg = float(lambda_sigreg)
        self.lambda_codebook = float(lambda_codebook)
        self.sigreg = SIGReg(**(sigreg_kwargs or {})) if self.lambda_sigreg > 0 else None

    def forward(self, outputs: dict) -> dict:
        """Combine prediction, codebook, and isotropy terms into one loss."""
        losses = self.pred(outputs)
        losses["loss"] = losses["pred_loss"]
        if outputs.get("codebook") is not None:
            losses["codebook"] = outputs["codebook"]
            losses["loss"] = losses["loss"] + self.lambda_codebook * losses["codebook"]
        if self.sigreg is not None and self.lambda_sigreg > 0:
            # SIGReg sees disposable projections when the facade provides them.
            if outputs.get("projected") is not None:
                import torch as _torch

                vals = [self.sigreg.statistic(t.float())
                        for t in outputs["projected"].values()]
                losses["sigreg"] = _torch.stack(vals).mean()
            else:
                losses["sigreg"] = self.sigreg(outputs["latents"])
            losses["loss"] = losses["loss"] + self.lambda_sigreg * losses["sigreg"]
        return losses
