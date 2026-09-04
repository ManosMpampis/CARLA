"""Part-reconstruction loss: MSE plus shape divergence on proposals.

Shape term mirrors the math of keonlee9420/Soft-DTW-Loss (sdtw_cuda_loss.py):
forward Bellman recursion with gamma-softmin over (diag, left, down), the
same backward structure (expected alignment under the Gibbs distribution),
and the same `normalize` divergence mode
``out_xy - (out_xx + out_yy) / 2`` (Blondel 2010.08354, non-negative and
minimized iff the series are equal). Differences from that file, deliberate:
pure-torch implementation running on CPU and CUDA with no numba dependency
(theirs asserts CUDA-only and caps at 1024 steps/block); gradients flow
through the softmin recursion via autograd instead of a custom Function.
Runs on proposed parts only, keeping the O(n*m) cost tractable.
"""
import math

import torch
import torch.nn as nn


def _pairwise_sq(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean cost matrix between (N, D) and (M, D) -> (N, M)."""
    return (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)


def _soft_dtw_table(d: torch.Tensor, gamma: float,
                    bandwidth: int | None) -> torch.Tensor:
    """Batched soft-DTW forward table over cost matrices (B, N, M)."""
    b, n, m = d.shape
    inf = torch.tensor(math.inf, device=d.device, dtype=d.dtype)
    table = torch.full((b, n + 2, m + 2), math.inf,
                       device=d.device, dtype=d.dtype)
    table[:, 0, 0] = 0.0
    allowed = None
    if bandwidth is not None and bandwidth > 0:
        # Sakoe-Chiba band: cells with |i - j| > band stay infinite.
        ii, jj = torch.meshgrid(torch.arange(n, device=d.device),
                                torch.arange(m, device=d.device), indexing="ij")
        allowed = (ii - jj).abs() <= int(bandwidth)
    for i in range(1, n + 1):
        row_allowed = allowed[i - 1] if allowed is not None else None
        for j in range(1, m + 1):
            if row_allowed is not None and not row_allowed[j - 1]:
                continue
            prev = torch.stack([table[:, i - 1, j - 1],
                                table[:, i - 1, j],
                                table[:, i, j - 1]], dim=-1)
            # Gamma-softmin of the three predecessors (same as reference).
            softmin = -gamma * (torch.logsumexp(-prev / gamma, dim=-1))
            table[:, i, j] = d[:, i - 1, j - 1] + softmin
    _ = inf
    return table[:, -2, -2]


class SoftDTW(nn.Module):
    """Batched soft-DTW with divergence and bandwidth flags.

    Args:
        gamma: smoothing; gamma -> 0 recovers hard DTW.
        normalize: divergence mode ``xy - (xx + yy) / 2`` (reference flag).
        bandwidth: Sakoe-Chiba half-width; None disables pruning.
    """

    def __init__(self, gamma: float = 1.0, normalize: bool = False,
                 bandwidth: int | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.normalize = bool(normalize)
        self.bandwidth = bandwidth

    @staticmethod
    def _cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pairwise squared-Euclidean costs for (B, N, D) x (B, M, D)."""
        return (x.unsqueeze(2) - y.unsqueeze(1)).pow(2).sum(-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Soft-DTW per batch pair; returns (B,) distances."""
        out_xy = _soft_dtw_table(self._cost(x, y), self.gamma, self.bandwidth)
        if not self.normalize:
            return out_xy
        # Divergence mode mirrors the reference: stack and split trick
        # reduces here to the definitional correction (same math).
        out_xx = _soft_dtw_table(self._cost(x, x), self.gamma, self.bandwidth)
        out_yy = _soft_dtw_table(self._cost(y, y), self.gamma, self.bandwidth)
        return out_xy - 0.5 * (out_xx + out_yy)


def soft_dtw_divergence(a: torch.Tensor, b: torch.Tensor,
                        gamma: float = 0.01) -> torch.Tensor:
    """Divergence between single (N, D) and (M, D) series; scalar output."""
    module = SoftDTW(gamma=gamma, normalize=True)
    return module(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)


class ReconLoss(nn.Module):
    """H2/Aux-recon criterion: MSE + lambda * shape divergence per part."""

    def __init__(self, lambda_shape: float = 0.0, gamma: float = 0.01,
                 max_parts: int = 4, bandwidth: int | None = None):
        super().__init__()
        self.lambda_shape = float(lambda_shape)
        self.gamma = float(gamma)
        self.max_parts = int(max_parts)
        self.sdtw = SoftDTW(gamma=gamma, normalize=True, bandwidth=bandwidth)

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
                # (1, L, C) batch pair straight into the mirror module.
                seg_r = recon[i, :, idx].transpose(0, 1).unsqueeze(0)
                seg_t = target[i, :, idx].transpose(0, 1).unsqueeze(0)
                shape = shape + self.sdtw(seg_r, seg_t).squeeze(0)
                valid += 1
            shape = shape / max(valid, 1)
        out["shape"] = shape
        out["loss"] = out["mse"] + self.lambda_shape * shape
        return out
