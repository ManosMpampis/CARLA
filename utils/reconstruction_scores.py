"""Shared residual-to-score reduction for all reconstruction arms."""

import torch


SCORE_MODES = ("l1", "l2", "mse", "kl")


def normalize_score_mode(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized not in SCORE_MODES:
        raise ValueError(f"score_mode must be one of {SCORE_MODES}; got {mode!r}")
    return normalized


def reconstruction_score_map(recon: torch.Tensor, target: torch.Tensor,
                             mode: str = "l1", *, tau: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (B,W) channel reduction and (B,C,W) element scores.

    L1 = mean_C |error|, L2 = sqrt(mean_C error²), MSE = mean_C error².
    A window's scalar score is always the mean over W of the first output.
    """
    mode = normalize_score_mode(mode)
    if recon.shape != target.shape or recon.ndim != 3:
        raise ValueError("prediction and target must have the same (B,C,T) shape")
    if mode == "kl":
        if not torch.isfinite(torch.tensor(tau)) or tau <= 0:
            raise ValueError("KL temperature must be finite and positive")
        log_p = torch.log_softmax(recon.float() / tau, dim=1)
        log_q = torch.log_softmax(target.float() / tau, dim=1)
        elements = 0.5 * (log_p.exp() - log_q.exp()) * (log_p - log_q)
        return elements.sum(dim=1), elements
    residual = recon.float() - target.float()
    if mode == "mse":
        elements = residual.square()
        return elements.mean(dim=1), elements
    elements = residual.abs()
    if mode == "l2":
        # vector_norm defines a finite zero subgradient at exact equality.
        return torch.linalg.vector_norm(residual, dim=1) / residual.size(1) ** 0.5, elements
    return elements.mean(dim=1), elements
