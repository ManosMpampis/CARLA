"""True-LeWM criterion: masked MSE prediction + SIGReg anti-collapse.

Prediction term: mean squared error between predictor output and the
full-view target latents, counted on MASKED token positions only (dense
when no mask is present, e.g. validation). Gradients flow into BOTH
streams (end-to-end LeWM: no stop-gradient, no EMA teacher).

Regularizer: SIGReg statistic over the pooled context+target tokens so the
shared encoder cannot collapse both streams to a constant. Returns the
`pred_loss` key the Trainer validation path reads.
"""
import torch
import torch.nn as nn

from losses.sigreg import SIGReg


class TrueLeWMLoss(nn.Module):
    """Masked latent-MSE + lambda * SIGReg (two loss terms, LeWM-style)."""

    def __init__(self, lambda_sigreg: float = 0.1,
                 sigreg_kwargs: dict | None = None):
        super().__init__()
        self.lambda_sigreg = float(lambda_sigreg)
        self.sigreg = SIGReg(**(sigreg_kwargs or {}))

    def forward(self, outputs: dict) -> dict:
        context, targets, predicted = (
            outputs["context"], outputs["targets"], outputs["predicted"])
        mask = outputs.get("mask")
        level_losses = []
        losses = {}
        for name in targets:
            diff = (predicted[name][:, 0] - targets[name]) ** 2
            m = None
            if isinstance(mask, dict) and mask.get(name) is not None:
                m = mask[name].to(diff.dtype).to(diff.device)
                while m.ndim < diff.ndim:
                    m = m.unsqueeze(1)
                m = m.expand_as(diff)
            if m is not None:
                lvl = (diff * m).sum() / m.sum().clamp(min=1.0)
            else:
                lvl = diff.mean()
            losses[f"pred_{name}"] = lvl
            level_losses.append(lvl)
        losses["pred_loss"] = torch.stack(level_losses).mean()
        if outputs.get("projected") is not None:
            # T2: disposable projections feed SIGReg; scoring never sees them.
            vals = [self.sigreg.statistic(t.float())
                    for t in outputs["projected"].values()]
            losses["sigreg"] = torch.stack(vals).mean()
        else:
            pooled = {n: torch.cat([context[n], targets[n]], dim=-1)
                      for n in targets}
            losses["sigreg"] = self.sigreg(pooled)
        losses["loss"] = losses["pred_loss"] + self.lambda_sigreg * losses["sigreg"]
        return losses
