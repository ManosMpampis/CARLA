"""Shared residual-to-score reduction for all reconstruction arms."""

import torch


SCORE_MODES = ("l1", "l2", "mse")


def normalize_score_mode(mode: str) -> str:
    normalized = str(mode).lower()
    if normalized not in SCORE_MODES:
        raise ValueError(f"score_mode must be one of {SCORE_MODES}; got {mode!r}")
    return normalized


def reconstruction_score_map(recon: torch.Tensor, target: torch.Tensor,
                             mode: str = "l1") -> tuple[torch.Tensor, torch.Tensor]:
    """Return (B,W) channel reduction and (B,C,W) element scores.

    L1 = mean_C |error|, L2 = sqrt(mean_C error²), MSE = mean_C error².
    A window's scalar score is always the mean over W of the first output.
    """
    mode = normalize_score_mode(mode)
    residual = recon.float() - target.float()
    if mode == "mse":
        elements = residual.square()
        return elements.mean(dim=1), elements
    elements = residual.abs()
    if mode == "l2":
        return residual.square().mean(dim=1).sqrt(), elements
    return elements.mean(dim=1), elements
