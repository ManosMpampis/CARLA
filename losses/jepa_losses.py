import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.sigreg import SIGReg


class JEPALoss(nn.Module):
    """Dense latent-prediction criterion.

    L1 between predicted and stop-gradient target latents over ALL tokens,
    levels and horizons (context positions included, V-JEPA 2.1 rationale),
    plus ``lambda_sigreg`` * SIGReg on the online latents. When a stage-A
    mask is supplied, prediction terms count only target positions that
    are masked; SIGReg always sees the full latents.
    """

    def __init__(self, lambda_sigreg: float = 0.0, sigreg_kwargs: dict | None = None,
                 target_norm: str | None = None, level_weights: dict | None = None,
                 lambda_codebook: float = 1.0):
        super().__init__()
        self.lambda_sigreg = float(lambda_sigreg)
        self.lambda_codebook = float(lambda_codebook)
        self.sigreg = SIGReg(**(sigreg_kwargs or {})) if self.lambda_sigreg > 0 else None
        self.target_norm = target_norm
        self.level_weights = level_weights or {}

    def _masked_target_positions(self, name, targets, mask):
        if mask is None or name not in mask:
            return None
        m = mask[name]
        if m.ndim == 1:
            m = m.unsqueeze(0).expand(targets.size(0), -1)
        return m  # (B, T_l) bool

    def forward(self, outputs: dict) -> dict:
        latents, targets, predicted, mask = (
            outputs["latents"], outputs["targets"], outputs["predicted"], outputs["mask"],
        )
        losses = {}
        level_losses = []
        for name in latents:
            tgt = targets[name]
            if self.target_norm == "layer":
                tgt = F.layer_norm(tgt, tgt.shape[1:])
            pred = predicted[name]
            b, k_total, d, t = pred.shape
            tmask = self._masked_target_positions(name, tgt, mask)

            err_sum = pred.new_zeros(())
            weight_sum = pred.new_zeros(())
            for k in range(1, k_total + 1):
                n_future = t - k
                if n_future <= 0:
                    continue
                diff = (pred[:, k - 1, :, :n_future] - tgt[:, :, k:]).abs()
                w = torch.ones(b, n_future, device=diff.device, dtype=diff.dtype)
                if tmask is not None:
                    w = w * tmask[:, k:].to(diff.dtype)
                err_sum = err_sum + (diff * w.unsqueeze(1)).sum()
                weight_sum = weight_sum + w.sum()

            level_loss = err_sum / weight_sum.clamp(min=1.0)
            losses[f"pred_{name}"] = level_loss
            level_losses.append(level_loss * self.level_weights.get(name, 1.0))

        losses["pred_loss"] = torch.stack(level_losses).mean()
        losses["loss"] = losses["pred_loss"]

        if outputs.get("codebook") is not None:
            losses["codebook"] = outputs["codebook"]
            losses["loss"] = losses["loss"] + self.lambda_codebook * losses["codebook"]

        if self.sigreg is not None and self.lambda_sigreg > 0:
            losses["sigreg"] = self.sigreg(latents)
            losses["loss"] = losses["loss"] + self.lambda_sigreg * losses["sigreg"]
        return losses
