import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (sliced Epps-Pulley).

    Latent tokens are projected onto a fixed set of random unit slices;
    per slice the empirical characteristic function is compared with the
    standard-normal one over a quadrature grid of frequencies. A constant
    (collapsed) latent maximizes the statistic, an isotropic Gaussian
    minimizes it. Projections are deterministic given the latent dim, so
    runs are reproducible and the buffer travels inside checkpoints.
    """

    def __init__(self, num_slices: int = 16, freq_nodes: int = 8,
                 freq_min: float = 0.2, freq_max: float = 4.0, seed: int = 4):
        super().__init__()
        self.num_slices = num_slices
        self.register_buffer(
            "freqs",
            torch.linspace(freq_min, freq_max, freq_nodes),
            persistent=False,
        )
        self._seed = seed
        self._slices = {}

    def _get_slices(self, dim: int, device, dtype) -> torch.Tensor:
        if dim not in self._slices:
            generator = torch.Generator().manual_seed(self._seed + dim)
            directions = torch.randn(self.num_slices, dim, generator=generator)
            directions = directions / directions.norm(dim=1, keepdim=True)
            self._slices[dim] = directions.to(device=device, dtype=dtype)
        return self._slices[dim]

    def statistic(self, tokens: torch.Tensor) -> torch.Tensor:
        """Epps-Pulley statistic for token embeddings of shape (N, D)."""
        slices = self._get_slices(tokens.size(1), tokens.device, tokens.dtype)
        z = tokens @ slices.t()  # (N, S)
        z = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-6)
        z = z.to(torch.float64)
        angles = z.unsqueeze(-1) * self.freqs.double()  # (N, S, F)
        phi_hat = torch.polar(torch.ones_like(angles), angles).mean(dim=0).abs() ** 2
        target = torch.exp(-(self.freqs.double() ** 2))
        return ((phi_hat - target) ** 2).sum(dim=1).mean()

    def forward(self, latents: dict) -> torch.Tensor:
        """Mean statistic over pyramid levels; latents maps level -> (B, D, T)."""
        values = []
        for name, z in latents.items():
            b, d, t = z.shape
            tokens = z.transpose(1, 2).reshape(b * t, d)
            values.append(self.statistic(tokens))
        return torch.stack(values).mean()
