"""Dense part-prediction loss (L1 over all tokens/offsets or masked parts).

V-JEPA 2.1 rationale: every token carries a target so per-sub-window
errors are well-defined at inference. Stage-A masks restrict the
prediction terms to masked positions; the regularizer always sees all.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DensePartLoss(nn.Module):
    """Mean L1 between predicted and stop-gradient target latents."""

    def __init__(self, level_weights: dict | None = None,
                 target_norm: str | None = None):
        super().__init__()
        self.level_weights = level_weights or {}
        self.target_norm = target_norm

    def forward(self, outputs: dict) -> dict:
        """Compute per-level and mean prediction loss from facade outputs."""
        latents, targets, predicted, mask = (
            outputs["latents"], outputs["targets"], outputs["predicted"], outputs["mask"])
        losses, level_losses = {}, []
        for name in latents:
            tgt = targets[name]
            if self.target_norm == "layer":
                tgt = F.layer_norm(tgt, tgt.shape[1:])
            pred = predicted[name]
            b, _, _, t = pred.shape
            m = mask.get(name) if isinstance(mask, dict) else None
            if m is not None and m.ndim == 1:
                m = m.unsqueeze(0).expand(b, -1)
            err, weight = pred.new_zeros(()), pred.new_zeros(())
            for k in range(1, pred.size(1) + 1):
                if t - k <= 0:
                    continue
                diff = (pred[:, k - 1, :, :t - k] - tgt[:, :, k:]).abs()
                w = torch.ones(b, t - k, device=diff.device, dtype=diff.dtype)
                if m is not None:
                    w = w * m[:, k:].to(diff.dtype)
                err = err + (diff * w.unsqueeze(1)).sum()
                weight = weight + w.sum()
            lvl = err / weight.clamp(min=1.0)
            losses[f"pred_{name}"] = lvl
            level_losses.append(lvl * self.level_weights.get(name, 1.0))
        losses["pred_loss"] = torch.stack(level_losses).mean()
        return losses
