"""Part-reconstruction loss: MSE plus shape divergence on proposals.

MSE catches spikes; Soft-DTW divergence (Blondel 2010.08354) catches
morphology under small shifts. The divergence (not raw Soft-DTW) is used:
non-negative and minimized iff series are equal. Runs on proposed parts
only, keeping the O(n*m) cost tractable. Short parts fall back to MSE.
"""
import torch
import torch.nn as nn


def _pairwise_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean cost matrix between (n, d) and (m, d)."""
    return (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)


def soft_dtw_divergence(a: torch.Tensor, b: torch.Tensor,
                        gamma: float = 0.01) -> torch.Tensor:
    """Soft-DTW divergence D(a,b) = sDTW(a,b) - (sDTW(a,a)+sDTW(b,b))/2.

    Minimal Bellman recursion with softmin (-gamma*logsumexp(-v/gamma)).
    Inputs are (n, d) and (m, d); returns a scalar. Differentiable.
    """

    def sdtw(x, y):
        d = _pairwise_sq(x, y)
        n, m = d.shape
        # Dynamic programming table with +inf borders.
        table = d.new_full((n + 2, m + 2), float("inf"))
        table[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                prev = torch.stack([table[i - 1, j], table[i, j - 1], table[i - 1, j - 1]])
                table[i, j] = d[i - 1, j - 1] - gamma * torch.logsumexp(-prev / gamma, dim=0)
        return table[n, m]

    return sdtw(a, b) - 0.5 * (sdtw(a, a) + sdtw(b, b))


class ReconLoss(nn.Module):
    """H2/Aux-recon criterion: MSE + lambda * shape divergence per part."""

    def __init__(self, lambda_shape: float = 0.0, gamma: float = 0.01,
                 max_parts: int = 4):
        super().__init__()
        self.lambda_shape = float(lambda_shape)
        self.gamma = float(gamma)
        self.max_parts = int(max_parts)

    def forward(self, recon: torch.Tensor, target: torch.Tensor,
                masks=None) -> dict:
        """Compare (B, C, W) recon vs target inside binary part masks."""
        if masks is not None:
            # Broadcast-aware mean: normalize by the actual masked elements.
            denom = masks.expand_as(recon).sum().clamp(min=1.0)
            mse = ((recon - target) ** 2 * masks).sum() / denom
        else:
            mse = ((recon - target) ** 2).mean()
        out = {"mse": mse}
        shape = recon.new_zeros(())
        if self.lambda_shape > 0 and masks is not None:
            valid = 0
            for i in range(min(recon.size(0), self.max_parts)):
                idx = (masks[i, 0] > 0.5).nonzero().flatten()
                if len(idx) < 2:
                    continue
                seg_r = recon[i, :, idx].transpose(0, 1)
                seg_t = target[i, :, idx].transpose(0, 1)
                shape = shape + soft_dtw_divergence(seg_r, seg_t, self.gamma)
                valid += 1
            shape = shape / max(valid, 1)
        out["shape"] = shape
        out["loss"] = out["mse"] + self.lambda_shape * shape
        return out
