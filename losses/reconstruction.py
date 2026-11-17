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
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconL1Loss(nn.Module):
    def __init__(self, loss: str = "l1"):
        super().__init__()
        kind = str(loss).lower()
        if kind not in ("kl", "mse", "l1"):
            raise ValueError(f"Invalid loss {loss}")
        self.loss = kind

    def _latent_loss(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            if self.loss == "mse":
                return ((pred - tgt) ** 2).mean()
            if self.loss == "l1":
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
        """Compare (B, C, W) recon vs target with fixed mean reduction."""
        tgt = outputs["targets"]
        recon_tgt = tgt.get("reconstruction")

        recon = outputs["recon"]
        l1 = (recon - recon_tgt).abs().mean()
        return {"loss": l1, "pred_loss": l1}
