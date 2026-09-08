"""Steered-LeWM criterion: configurable latent loss + aux mask loss + dual SIGReg.

Loss  = w_pred * Loss2(Z', Z) + w_aux * MSE(sigmoid(Time_mask), mask)
      + lambda_sigreg * SIGReg(Z) + lambda_sigreg_tgt * SIGReg(Z').

Loss2 kinds: kl (default, symmetric channel-wise KL without detach),
mse, l1. No stop-gradient anywhere by design; both streams train end to end.
Returns the `pred_loss` key the Trainer validation path reads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.sigreg import SIGReg


class SteeredLeWMLoss(nn.Module):
    """Two-head LeWM loss with selectable latent geometry."""

    def __init__(self, loss2_kind: str = "kl", w_pred: float = 1.0,
                 w_aux: float = 1.0, lambda_sigreg: float = 0.1,
                 lambda_sigreg_tgt: float = 0.1, tau: float = 1.0,
                 sigreg_kwargs: dict | None = None):
        super().__init__()
        kind = str(loss2_kind).lower()
        if kind not in ("kl", "mse", "l1"):
            raise ValueError(f"Invalid loss2_kind {loss2_kind}")
        self.loss2_kind = kind
        self.w_pred = float(w_pred)
        self.w_aux = float(w_aux)
        self.lambda_sigreg = float(lambda_sigreg)
        self.lambda_sigreg_tgt = float(lambda_sigreg_tgt)
        self.tau = float(tau)
        self.sigreg = SIGReg(**(sigreg_kwargs or {}))

    def _latent_loss(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if self.loss2_kind == "mse":
            return ((pred - tgt) ** 2).mean()
        if self.loss2_kind == "l1":
            return (pred - tgt).abs().mean()
        tau = max(self.tau, 1e-3)
        log_p = F.log_softmax(pred / tau, dim=1)
        log_q = F.log_softmax(tgt / tau, dim=1)
        p = log_p.exp()
        q = log_q.exp()
        kl_pq = (p * (log_p - log_q)).sum(dim=1).mean()
        kl_qp = (q * (log_q - log_p)).sum(dim=1).mean()
        return 0.5 * (kl_pq + kl_qp)

    def forward(self, outputs: dict) -> dict:
        """Compute total loss from steered-model outputs."""
        tgt = outputs["targets"]["L0"]
        pred = outputs["predicted"]["L0"]
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred[:, 0]
        pred_loss = self._latent_loss(pred, tgt)
        logits = outputs["mask_logits"]
        m = outputs.get("mask_target")
        if m is None and isinstance(outputs.get("mask"), dict):
            m = outputs["mask"].get("input")
        if m is None:
            aux_loss = logits.new_zeros(())
        else:
            aux_loss = F.mse_loss(torch.sigmoid(logits),
                                  m.to(logits.dtype).float())
        sig = self.sigreg({"L0": tgt})
        sig_tgt = self.sigreg({"L0": pred})
        loss = self.w_pred * pred_loss + self.w_aux * aux_loss \
            + self.lambda_sigreg * sig + self.lambda_sigreg_tgt * sig_tgt
        return {"loss": loss, "pred_loss": pred_loss, "aux_loss": aux_loss,
                "sigreg": sig, "sigreg_tgt": sig_tgt}
