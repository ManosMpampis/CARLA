"""LeWM facade: encoder + part-predictors + anti-collapse, one object.

Clean-room rewrite of the old JEPA core with the same external contract
(encode/predict/forward/score) so Trainer, Scorer, and configs keep working.
New: disposable projector before SIGReg, optional action conditioning
(closed-loop arm), configurable predictor wiring.
"""
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conditioner import ActionEmbed, Projector
from models.ema import EMAWrapper
from models.predictor import CondPredictor, build_predictor

PREDICTOR_REGISTRY = ("masked", "tcn", "gru")
ANTI_COLLAPSE_REGISTRY = ("none", "sigreg", "ema", "codebook")


class LeWMModel(nn.Module):
    """LeWM trunk: shared encoder pyramid + one part-predictor per level."""

    def __init__(self, encoder: nn.Module, predictor: str = "masked",
                 horizons: int = 2, predictor_hidden=None,
                 predictor_kwargs: dict | None = None,
                 anti_collapse: str = "none", ema_momentum: float = 0.99925,
                 codebook_kwargs: dict | None = None, target_norm=None,
                 use_projector: bool = False, projector_hidden=None,
                 action_dim: int = 0):
        super().__init__()
        if predictor not in PREDICTOR_REGISTRY:
            raise ValueError(f"Invalid predictor {predictor}")
        if anti_collapse not in ANTI_COLLAPSE_REGISTRY:
            raise ValueError(f"Invalid anti_collapse {anti_collapse}")
        self.encoder = encoder
        names = cast("list[str]", getattr(encoder, "level_names"))
        dims = cast("list[int]", getattr(encoder, "level_dims"))
        strides = cast("list[int]", getattr(encoder, "level_strides", [1] * len(names)))
        self.level_names, self.level_dims = list(names), [int(d) for d in dims]
        self.level_strides = [int(s) for s in strides]

        # One predictor per level; wrapped for action conditioning if asked.
        self.predictors = nn.ModuleDict({
            n: build_predictor(predictor, d, horizons, predictor_hidden,
                               **(predictor_kwargs or {}))
            for n, d in zip(self.level_names, self.level_dims)})
        self.action: ActionEmbed | None = None
        if action_dim and int(action_dim) > 0:
            self.action = ActionEmbed(action_dim=int(action_dim))
            for n, d in zip(self.level_names, self.level_dims):
                self.predictors[n] = CondPredictor(
                    self.predictors[n], d, action_dim=int(action_dim))
        self.horizons = horizons
        self.anti_collapse = anti_collapse
        self.target_norm = target_norm or ("layer" if anti_collapse == "ema" else None)

        # Disposable projector: SIGReg sees projections, scoring never does.
        self.projectors: nn.ModuleDict | None = None
        if use_projector:
            self.projectors = nn.ModuleDict({
                n: Projector(d, hidden=projector_hidden)
                for n, d in zip(self.level_names, self.level_dims)})

        self.target_encoder = None
        self.codebook = None
        self.encoder_frozen = False
        if anti_collapse == "ema":
            self.target_encoder = EMAWrapper(encoder, momentum=ema_momentum)
        elif anti_collapse == "codebook":
            from models.codebook import SoftCodebook
            self.codebook = SoftCodebook(dims=self.level_dims,
                                         **(codebook_kwargs or {}))

    # -- forward pieces -------------------------------------------------
    def encode(self, x: torch.Tensor) -> dict:
        """Encode a window into per-level latents."""
        return self.encoder(x)

    def predict(self, latents: dict, action=None, masks=None) -> dict:
        """Predict part latents, optionally conditioned on action.

        Masked predictors receive their level mask (True = predict this
        token); causal predictors ignore masks (uniform signature).
        """
        masks = masks or {}
        out = {}
        for n in latents:
            pred = self.predictors[n]
            m = masks.get(n) if isinstance(masks, dict) else None
            if isinstance(pred, CondPredictor):
                out[n] = pred(latents[n], action, m)
            else:
                out[n] = pred(latents[n], m)
        return out

    @torch.no_grad()
    def _targets_from(self, x: torch.Tensor, latents: dict) -> dict:
        if self.target_encoder is not None:
            return self.target_encoder.encode(x)
        return {n: z.detach() for n, z in latents.items()}

    def forward(self, x: torch.Tensor, mask: dict | None = None,
                action=None) -> dict:
        """Encode, form stop-gradient targets, predict; ready for criterion."""
        latents = self.encode(x)
        outputs: dict[str, object] = {
            "latents": latents,
            "targets": self._targets_from(x, latents),
            "predicted": self.predict(latents, action, mask),
            "mask": mask,
        }
        if self.codebook is not None:
            outputs["codebook"] = self.codebook.quantization_loss(latents)
        if self.projectors is not None:
            outputs["projected"] = {n: self.projectors[n](
                latents[n].transpose(1, 2).reshape(-1, latents[n].size(1)))
                for n in latents}
        return outputs

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load weights, transparently remapping pre-rebuild key names.

        Any direct load of an old checkpoint keeps working at every call
        site; see remap_legacy_encoder_keys for the mapping.
        """
        from models.encoder import remap_legacy_encoder_keys

        return super().load_state_dict(
            remap_legacy_encoder_keys(state_dict), strict=strict, assign=assign)

    def update_ema(self) -> None:
        """Advance the trailing teacher after each optimizer step."""
        if self.target_encoder is not None:
            self.target_encoder.update(self.encoder)

    def update_running_stats(self, latents: dict) -> None:
        """Per-step running statistics hook; no-op on the bare trunk."""

    def latent_variance(self, latents: dict) -> float:
        """Collapse diagnostic: mean per-dim variance across levels."""
        vals = [latents[n].transpose(1, 2).reshape(-1, latents[n].size(1)).var(dim=0).mean()
                for n in latents]
        return torch.stack(vals).mean().item()

    # -- scoring (fp32, eval mode; never touches projections) ------------
    @torch.no_grad()
    def score(self, x: torch.Tensor) -> dict:
        """Per-window anomaly evidence fused across levels to (B, W)."""
        was_training = self.training
        self.eval()
        latents = self.encode(x)
        predicted = self.predict(latents)
        targets = self._targets_from(x, latents)
        b, _, window = x.shape
        sums = x.new_zeros((b, window), dtype=torch.float32)
        level_maps: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self.level_names):
            stride = self.level_strides[idx]
            steps = self._token_errors(predicted[name], targets[name])
            steps = steps.repeat_interleave(stride, dim=1)[:, :window]
            sums = sums + steps
            level_maps[name] = steps
        fused = sums / max(len(self.level_names), 1)
        signals_out: dict[str, torch.Tensor] = {}
        out = {"fused": fused, "levels": level_maps, "signals": signals_out}
        if self.codebook is not None:
            for name in latents:
                dist, entropy = self.codebook.signals(name, latents[name])
                stride = self.level_strides[self.level_names.index(name)]
                signals_out[f"{name}/codebook_dist"] = dist.repeat_interleave(stride, dim=1)[:, :window]
                signals_out[f"{name}/attn_entropy"] = entropy.repeat_interleave(stride, dim=1)[:, :window]
        if was_training:
            self.train()
        return out

    def _token_errors(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean |pred - target| per future/part position over valid offsets."""
        k_total, length = pred.size(1), target.size(-1)
        err_sum = None
        for k in range(1, k_total + 1):
            n_future = length - k
            if n_future <= 0:
                continue
            diff = (pred[:, k - 1, :, :n_future] - target[:, :, k:]).abs().mean(dim=1)
            padded = F.pad(diff, (k, 0))
            err_sum = padded if err_sum is None else err_sum + padded
        assert err_sum is not None, "window shorter than the first offset"
        weights = torch.minimum(torch.arange(length, device=target.device) + 1,
                                torch.tensor(k_total, device=target.device)).clamp(min=1)
        return err_sum / weights


class FullLeWMModel(nn.Module):
    """Full-training wrapper: trunk plus attached aux/H heads in one module.

    Honors the Trainer contract by delegating level geometry, teacher and
    codebook handles, and scoring to the trunk, so the existing stage
    runner works unchanged. Forward runs the trunk once and evaluates only
    the attached heads; the FullCriterion consumes the merged outputs.
    Box-supervised and proposal-masked terms stay dormant until the batch
    carries 'box_target' / 'part_masks' (proposal wiring, staged later).
    """

    def __init__(self, trunk: LeWMModel, in_channels: int,
                 recon_aux=None, box_aux=None, heads: dict | None = None,
                 center_momentum: float = 0.99):
        super().__init__()
        self.trunk = trunk
        self.recon_aux = recon_aux
        self.box_aux = box_aux
        self.heads = nn.ModuleDict(heads or {})
        self.center_momentum = float(center_momentum)
        self.in_channels = int(in_channels)

    # -- Trainer contract (delegated to the trunk) ----------------------
    @property
    def level_names(self):
        """Pyramid level names, delegated for collator/scorer geometry."""
        return self.trunk.level_names

    @property
    def level_strides(self):
        """Cumulative strides, delegated for mask and score mapping."""
        return self.trunk.level_strides

    @property
    def target_encoder(self):
        """Trailing teacher handle, delegated (None unless EMA arm)."""
        return self.trunk.target_encoder

    @property
    def codebook(self):
        """Codebook handle, delegated (None unless codebook arm)."""
        return self.trunk.codebook

    @property
    def encoder_frozen(self):
        """Frozen-adaptation flag, delegated to the trunk."""
        return self.trunk.encoder_frozen

    @property
    def anti_collapse(self):
        """Anti-collapse name, delegated for checkpoint metadata."""
        return self.trunk.anti_collapse

    def update_ema(self) -> None:
        """Advance the trailing teacher; delegated to the trunk."""
        self.trunk.update_ema()

    def latent_variance(self, latents: dict) -> float:
        """Collapse diagnostic; delegated to the trunk."""
        return self.trunk.latent_variance(latents)

    def score(self, x: torch.Tensor) -> dict:
        """Anomaly evidence; scoring never touches heads or projections."""
        return self.trunk.score(x)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load weights, remapping legacy trunk keys like the trunk does."""
        from models.encoder import remap_legacy_encoder_keys

        return super().load_state_dict(
            remap_legacy_encoder_keys(state_dict), strict=strict, assign=assign)

    def trunk_state_dict(self) -> dict:
        """Trunk-only weights, loadable into LeWMModel for scoring reuse."""
        prefix = "trunk."
        return {k[len(prefix):]: v for k, v in self.state_dict().items()
                if k.startswith(prefix)}

    def update_running_stats(self, latents: dict) -> None:
        """Per-step running centers for the attached metric head, if any."""
        if "h4" in self.heads:
            self.heads["h4"].update_centers(latents, self.center_momentum)

    # -- forward ---------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: dict | None = None,
                action=None) -> dict:
        """Run trunk once plus attached heads; merged outputs for criterion."""
        trunk_out = self.trunk(x, mask=mask, action=action)
        feats = trunk_out["latents"]
        window = x.size(-1)
        out: dict[str, object] = {
            "latents": feats,
            "mask": mask,
            "trunk": trunk_out,
            "window": x,
        }
        if self.recon_aux is not None:
            out["aux_recon"] = self.recon_aux(feats, window)
        if self.box_aux is not None:
            coarse = max(feats, key=lambda n: feats[n].shape[1])
            out["aux_boxes"] = self.box_aux(feats[coarse])
        if "h1" in self.heads:
            out["h1"] = self.heads["h1"](feats)
        if "h2" in self.heads:
            out["h2"] = self.heads["h2"](feats, window)
        if "h3" in self.heads:
            out["h3"] = self.heads["h3"](feats)
        if "h4" in self.heads:
            head = self.heads["h4"]
            out["h4"] = head(feats)
            out["h4_raw"] = head.raw(feats)
            out["h4_centers"] = head.centers()
        return out
