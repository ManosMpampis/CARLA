"""Sketched isotropic Gaussian regularization for LeWM latents.

The Epps--Pulley statistic follows the LeWM reference implementation:
https://github.com/lucas-maes/le-wm/blob/main/module.py
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Match random projections of encoder latents to a standard normal.

    ``statistic`` accepts (N, D) or (..., N, D) embeddings. ``forward``
    accepts pyramid levels of shape (B, D, T) and tests each time step
    across the batch, then averages over time and levels.
    """

    def __init__(self, num_slices: int = 16, freq_nodes: int = 17,
                 freq_min: float = 0.0, freq_max: float = 3.0,
                 seed: int | None = None):
        super().__init__()
        if num_slices < 1 or freq_nodes < 2:
            raise ValueError("num_slices must be positive and freq_nodes at least 2")
        if freq_min < 0 or freq_max <= freq_min:
            raise ValueError("frequency range must satisfy 0 <= min < max")
        self.num_slices = num_slices
        self.seed = seed
        self._generators: dict[torch.device, torch.Generator] = {}

        freqs = torch.linspace(freq_min, freq_max, freq_nodes)
        step = (freq_max - freq_min) / (freq_nodes - 1)
        weights = torch.full((freq_nodes,), 2 * step)
        weights[[0, -1]] = step
        phi = torch.exp(-0.5 * freqs.square())
        self.register_buffer("freqs", freqs, persistent=False)
        self.register_buffer("phi", phi, persistent=False)
        self.register_buffer("weights", weights * phi, persistent=False)

    def _directions(self, dim: int, device: torch.device) -> torch.Tensor:
        generator = None
        if self.seed is not None:
            if device not in self._generators:
                self._generators[device] = torch.Generator(device=device).manual_seed(self.seed)
            generator = self._generators[device]
        directions = torch.randn(dim, self.num_slices, device=device,
                                 generator=generator)
        return directions / directions.norm(dim=0, keepdim=True)

    def statistic(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mean Epps--Pulley statistic over sketches and leading dimensions."""
        if tokens.ndim < 2 or tokens.size(-2) < 1:
            raise ValueError("tokens must have shape (..., N, D) with N > 0")
        # Keep the characteristic-function calculation in fp32 under AMP.
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            x = tokens.float()
            directions = self._directions(x.size(-1), x.device)
            angles = (x @ directions).unsqueeze(-1) * self.freqs.to(x.device)
            cosine = angles.cos().mean(dim=-3)
            sine = angles.sin().mean(dim=-3)
            error = (cosine - self.phi.to(x.device)).square() + sine.square()
            per_slice = (error @ self.weights.to(x.device)) * x.size(-2)
            return per_slice.mean()

    def forward(self, latents: dict[str, torch.Tensor]) -> torch.Tensor:
        """Average the batch-wise statistic across time and pyramid levels."""
        values = []
        for z in latents.values():
            if z.ndim != 3:
                raise ValueError("each latent level must have shape (B, D, T)")
            values.append(self.statistic(z.permute(2, 0, 1)))
        return torch.stack(values).mean()
