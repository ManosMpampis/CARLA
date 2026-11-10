"""Matched from-scratch reconstruction arms for the LeWM comparison."""

import torch
from torch import nn

from models.recon_head import build_mirrored_head
from models.feature_paths import FeaturePath
from utils.model_contract import ModelContract
from utils.reconstruction_scores import reconstruction_score_map, normalize_score_mode


class ReconstructionBaseline(ModelContract, nn.Module):
    """The same encoder/decoder in both arms; only VAE adds a posterior."""

    def __init__(self, model_kwargs: dict, recon_kwargs: dict, variational: bool,
                 score_mode: str = "l1", encoder=None, backbone=None,
                 freeze_encoder=False, freeze_backbone=True):
        super().__init__()
        if encoder is None:
            from models import get_backbone

            encoder = get_backbone("lewm_resnet", **model_kwargs)["model"]
        self.encoder = encoder
        self.backbone = backbone
        self.encoder_frozen = bool(freeze_encoder)
        self.backbone_frozen = bool(backbone is not None and freeze_backbone)
        if self.encoder_frozen:
            self.encoder.requires_grad_(False).eval()
        if self.backbone_frozen:
            self.backbone.requires_grad_(False).eval()
        features = FeaturePath([backbone, encoder]) if backbone is not None else encoder
        self.decoder = build_mirrored_head(
            features,
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
        return self.encoder.total_stride * getattr(self.backbone, "total_stride", 1)

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def encode(self, x):
        if x.size(-1) < self.total_stride or x.size(-1) % self.total_stride:
            raise ValueError("input length must be a positive multiple of feature stride")
        return self.encoder(self.backbone(x) if self.backbone is not None else x)

    def forward(self, x: torch.Tensor, sample: bool | None = None, *, mask=None, action=None) -> dict:
        if mask is not None or action is not None:
            raise ValueError("AE and VAE do not use masks or actions")
        h = self.encode(x)
        if self.variational:
            mu = self.mu(h)
            logvar = self.logvar(h).clamp(-20.0, 20.0)
            sample = self.training if sample is None else sample
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
            latent = mu
        else:
            mu = logvar = None
            z = latent = h
        return {"recon": self.decoder(z), "target": x, "mu": mu, "logvar": logvar,
                "latent": latent, "latents": latent}

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> dict:
        """One configured reconstruction-error score per timestep."""
        was_training = self.training
        self.eval()
        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                reconstruction = self.forward(x.float(), sample=False)["recon"]
                fused, error = reconstruction_score_map(reconstruction, x, self.score_mode)
            return {"fused": fused,
                    "signals": {f"channel/{c}": error[:, c] for c in range(error.shape[1])}}
        finally:
            self.train(was_training)


def reconstruction_objective(outputs: dict, target: torch.Tensor,
                             beta: float) -> dict:
    recon = outputs["recon"]
    if recon.shape != target.shape:
        raise ValueError(f"reconstruction {recon.shape} must match target {target.shape}")
    mse = (recon.float() - target.float()).square().mean()
    if outputs["mu"] is None:
        kl = mse.new_zeros(())
    else:
        mu = outputs["mu"].float()
        logvar = outputs["logvar"].float()
        kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
    return {"loss": mse + float(beta) * kl, "mse": mse, "kl": kl}
