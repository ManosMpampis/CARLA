from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ema import EMAWrapper


def build_predictor(kind, dim, horizons, hidden=None):
    if kind not in PREDICTOR_REGISTRY:
        raise ValueError("Invalid predictor {}".format(kind))
    return PREDICTOR_REGISTRY[kind](dim, horizons=horizons, hidden=hidden)


def _build_tcn(dim, horizons, hidden=None):
    from models.jepa_pyramid import CausalTCNPredictor

    return CausalTCNPredictor(dim, horizons=horizons, hidden=hidden)


def _build_gru(dim, horizons, hidden=None):
    from models.jepa_pyramid import GRUPredictor

    return GRUPredictor(dim, horizons=horizons, hidden=hidden)


PREDICTOR_REGISTRY = {"tcn": _build_tcn, "gru": _build_gru}
ANTI_COLLAPSE_REGISTRY = ("none", "sigreg", "ema", "codebook")


class JEPAModel(nn.Module):
    """JEPA facade: encoder pyramid + one causal predictor per level.

    encode(x)            -> {level: latents (B, D_l, T_l)}
    predict(latents)     -> {level: (B, horizons, D_l, T_l)} future predictions
    forward(x)           -> latents + stop-gradient targets + predictions,
                            ready for the JEPA criterion.
    score(x)             -> per-window anomaly evidence at sub-window
                            granularity (fp32, eval-time only).
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: str = "tcn",
        horizons: int = 2,
        predictor_hidden: int | None = None,
        anti_collapse: str = "none",
        ema_momentum: float = 0.99925,
        codebook_kwargs: dict | None = None,
        target_norm: str | None = None,
    ):
        super().__init__()
        if predictor not in PREDICTOR_REGISTRY:
            raise ValueError("Invalid predictor {}".format(predictor))
        if anti_collapse not in ANTI_COLLAPSE_REGISTRY:
            raise ValueError("Invalid anti_collapse {}".format(anti_collapse))

        self.encoder = encoder
        names = cast("list[str]", getattr(encoder, "level_names"))
        dims = cast("list[int]", getattr(encoder, "level_dims"))
        strides = cast("list[int]", getattr(encoder, "level_strides",
                                            [1] * len(names)))
        self.level_names: list[str] = list(names)
        self.level_dims: list[int] = [int(d) for d in dims]
        self.level_strides: list[int] = [int(s) for s in strides]

        self.predictors = nn.ModuleDict({
            name: build_predictor(predictor, dim, horizons, predictor_hidden)
            for name, dim in zip(self.level_names, self.level_dims)
        })
        self.horizons = horizons
        self.anti_collapse = anti_collapse
        self.target_norm = target_norm or ("layer" if anti_collapse == "ema" else None)

        self.target_encoder = None
        self.codebook = None
        self.encoder_frozen = False
        if anti_collapse == "ema":
            self.target_encoder = EMAWrapper(encoder, momentum=ema_momentum)
        elif anti_collapse == "codebook":
            from models.codebook import SoftCodebook

            self.codebook = SoftCodebook(dims=self.level_dims, **(codebook_kwargs or {}))

    # ------------------------------------------------------------------ #
    # forward pieces                                                     #
    # ------------------------------------------------------------------ #
    def encode(self, x: torch.Tensor) -> dict:
        return self.encoder(x)

    def predict(self, latents: dict) -> dict:
        return {name: self.predictors[name](latents[name]) for name in latents}

    @torch.no_grad()
    def _targets_from(self, x: torch.Tensor, latents: dict) -> dict:
        if self.target_encoder is not None:
            return self.target_encoder.encode(x)
        return {name: z.detach() for name, z in latents.items()}

    def forward(self, x: torch.Tensor, mask: dict | None = None) -> dict:
        latents = self.encode(x)
        targets = self._targets_from(x, latents)
        predicted = self.predict(latents)
        outputs: dict[str, object] = {
            "latents": latents,
            "targets": targets,
            "predicted": predicted,
            "mask": mask,
        }
        if self.codebook is not None:
            outputs["codebook"] = self.codebook.quantization_loss(latents)
        return outputs

    def update_ema(self) -> None:
        if self.target_encoder is not None:
            self.target_encoder.update(self.encoder)

    def latent_variance(self, latents: dict) -> float:
        """Collapse diagnostic: mean per-dim variance across levels."""
        values = []
        for z in latents.values():
            b, d, t = z.shape
            tokens = z.transpose(1, 2).reshape(b * t, d)
            values.append(tokens.var(dim=0).mean())
        return torch.stack(values).mean().item()

    # ------------------------------------------------------------------ #
    # scoring path (fp32, eval mode; never touches SIGReg projections)   #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score(self, x: torch.Tensor) -> dict:
        """Per-window anomaly evidence.

        Returns fused per-timestep scores (B, W): prediction error at each
        pyramid level is mapped onto the input steps its token covers and
        mean-fused across levels. Codebook arms additionally emit
        distance-to-prototype and attention-entropy signal maps.
        """
        was_training = self.training
        self.eval()
        latents = self.encode(x)
        predicted = self.predict(latents)
        # surprise is divergence from what the target branch encodes:
        # the trailing teacher on EMA arms, own latents (stop-grad) otherwise
        targets = self._targets_from(x, latents)

        b, _, window = x.shape
        sums = x.new_zeros((b, window), dtype=torch.float32)
        counts = 0
        level_maps: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self.level_names):
            stride = self.level_strides[idx]
            tok = self._token_errors(predicted[name], targets[name])  # (B, T_l)
            steps = tok.repeat_interleave(stride, dim=1)[:, :window]
            sums = sums + steps
            counts += 1
            level_maps[name] = steps
        fused = sums / max(counts, 1)
        signals_out: dict[str, torch.Tensor] = {}
        out = {"fused": fused, "levels": level_maps, "signals": signals_out}

        if self.codebook is not None:
            for name in latents:
                dist, entropy = self.codebook.signals(name, latents[name])
                stride = self.level_strides[self.level_names.index(name)]
                signals_out[f"{name}/codebook_dist"] = \
                    dist.repeat_interleave(stride, dim=1)[:, :window]
                signals_out[f"{name}/attn_entropy"] = \
                    entropy.repeat_interleave(stride, dim=1)[:, :window]
        if was_training:
            self.train()
        return out

    def _token_errors(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean |pred - target| per *future* token position over its valid horizons."""
        k_total = pred.size(1)
        length = target.size(-1)
        err_sum: torch.Tensor | None = None
        for k in range(1, k_total + 1):
            n_future = length - k
            if n_future <= 0:
                continue
            diff = (pred[:, k - 1, :, :n_future] - target[:, :, k:]).abs().mean(dim=1)
            padded = F.pad(diff, (k, 0))  # align error to the *future* position
            err_sum = padded if err_sum is None else err_sum + padded
        assert err_sum is not None, "window shorter than the first horizon"
        weights = torch.minimum(
            torch.arange(length, device=target.device) + 1,
            torch.tensor(k_total, device=target.device),
        ).clamp(min=1)
        return err_sum / weights
