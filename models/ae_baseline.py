"""Matched from-scratch reconstruction arms for the LeWM comparison."""

import torch
from torch import nn

from models.recon_head import build_mirrored_head
from utils.reconstruction_scores import reconstruction_score_map, normalize_score_mode


class ReconstructionBaseline(nn.Module):
    """The same encoder/decoder in both arms; only VAE adds a posterior."""

    def __init__(self, model_kwargs: dict, recon_kwargs: dict, variational: bool,
                 score_mode: str = "l1", encoder=None):
        super().__init__()
        if encoder is None:
            from models import get_backbone

            encoder = get_backbone("lewm_resnet", **model_kwargs)["model"]
        self.encoder = encoder
        self.decoder = build_mirrored_head(
            self.encoder,
            norm=recon_kwargs.get("norm", model_kwargs.get("norm", "batch")),
            dropout=float(recon_kwargs.get("dropout", model_kwargs.get("dropout", 0.0))),
        )
        self.variational = bool(variational)
        self.score_mode = normalize_score_mode(score_mode)
        if self.variational:
            d = self.encoder.output_dims
            self.mu = nn.Conv1d(d, d, kernel_size=1)
            self.logvar = nn.Conv1d(d, d, kernel_size=1)

    @property
    def total_stride(self) -> int:
        return self.encoder.total_stride

    def forward(self, x: torch.Tensor, sample: bool | None = None) -> dict:
        h = self.encoder(x)
        if not self.variational:
            return {"recon": self.decoder(h), "mu": None, "logvar": None}
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-20.0, 20.0)
        if sample is None:
            sample = self.training
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return {"recon": self.decoder(z), "mu": mu, "logvar": logvar}

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> dict:
        """One configured reconstruction-error score per timestep."""
        reconstruction = self.forward(x.float(), sample=False)["recon"]
        fused, error = reconstruction_score_map(
            reconstruction, x, self.score_mode)
        return {"fused": fused, "levels": {"L0": fused},
                "signals": {f"channel/{c}": error[:, c]
                            for c in range(error.shape[1])}}


def reconstruction_objective(outputs: dict, target: torch.Tensor,
                             beta: float) -> dict:
    mse = (outputs["recon"] - target).square().mean()
    if outputs["mu"] is None:
        kl = mse.new_zeros(())
    else:
        mu = outputs["mu"].float()
        logvar = outputs["logvar"].float()
        kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
    return {"loss": mse + float(beta) * kl, "mse": mse, "kl": kl}
