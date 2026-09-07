"""True-LeWM model: two-stream input masking over ONE shared encoder.

Stream 1 (context): X -> mask_input(X) -> Encoder -> z_ctx
                    -> Predictor(z_ctx, action) -> z_pred
Stream 2 (target):  X -> Encoder -> z_tgt   (same weights, second pass,
                    NO detach: end-to-end LeWM training, collapse is handled
                    by SIGReg in the criterion, not by stop-gradient/EMA).

The encoder is any pyramid encoder exposing level_names / level_dims /
level_strides (e.g. PyramidEncoder from models.encoder). Each level owns
one action-conditioned predictor: CondPredictor (zero-init AdaLN-style
scale/shift) around a MaskedReconPredictor used as a decoder (no latent
mask tokens here: the masking already happened in the INPUT space).

Action semantics: per-window [mask_ratio, mask_center, mask_span] triple
embedded by ActionEmbed. The action tells the predictor *where/what* to
reconstruct -- the I-JEPA positional-mask-token role in LeWM AdaLN form.
At scoring time the same channel becomes the probe query.
"""
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conditioner import ActionEmbed
from models.predictor import CondPredictor, MaskedReconPredictor


class TrueLeWMModel(nn.Module):
    """Shared-encoder two-stream world model for masked latent prediction."""

    def __init__(self, encoder: nn.Module, action_dim: int = 16,
                 predictor_kwargs: dict | None = None):
        super().__init__()
        self.encoder = encoder
        names = cast("list[str]", getattr(encoder, "level_names"))
        dims = cast("list[int]", getattr(encoder, "level_dims"))
        strides = cast("list[int]", getattr(encoder, "level_strides", [1] * len(names)))
        self.level_names, self.level_dims = list(names), [int(d) for d in dims]
        self.level_strides = [int(s) for s in strides]

        pk = dict(predictor_kwargs or {})
        self.predictors = nn.ModuleDict({
            n: CondPredictor(MaskedReconPredictor(d, **pk), d,
                             action_dim=int(action_dim))
            for n, d in zip(self.level_names, self.level_dims)})
        self.action = ActionEmbed(action_dim=int(action_dim))
        self.action_dim = int(action_dim)

        # Trainer contract handles (None unless an EMA/codebook arm).
        self.target_encoder = None
        self.codebook = None
        self.anti_collapse = "sigreg"
        self.encoder_frozen = False

    # -- masking helpers -------------------------------------------------
    @staticmethod
    def _mask_input(x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """Zero masked input steps (post-standardization zeros = channel mean)."""
        m = input_mask.to(torch.bool)
        while m.ndim < x.ndim:
            m = m.unsqueeze(1)
        return x.masked_fill(m.expand_as(x), 0.0)

    def _action_from_mask(self, input_mask: torch.Tensor) -> torch.Tensor:
        """Describe the mask as a (B, 3) triple for the ActionEmbed."""
        m = input_mask.to(torch.bool).float()
        b, w = m.shape
        idx = torch.arange(w, device=m.device, dtype=m.dtype)
        count = m.sum(dim=1).clamp(min=1.0)
        ratio = m.mean(dim=1)
        center = (m * idx).sum(dim=1) / count / float(w)
        first = torch.where(m.bool(), idx.unsqueeze(0).expand_as(m),
                            torch.full_like(m, float(w))).min(dim=1).values / float(w)
        last = torch.where(m.bool(), idx.unsqueeze(0).expand_as(m),
                           torch.full_like(m, -1.0)).max(dim=1).values.clamp(min=0) / float(w)
        span = (last - first).clamp(min=0.0)
        triple = torch.stack([ratio, center, span], dim=1)
        return self.action(triple)

    # -- forward pieces --------------------------------------------------
    def encode(self, x: torch.Tensor) -> dict:
        """Encode a window into per-level latents."""
        return self.encoder(x)

    def predict(self, latents: dict, action=None) -> dict:
        """Decode context latents toward target latents, conditioned on action."""
        return {n: self.predictors[n](latents[n], action, None) for n in latents}

    def forward(self, x: torch.Tensor, mask: dict | None = None,
                action=None) -> dict:
        """Two streams, one encoder; keys match the Trainer/criterion contract."""
        token_masks = {}
        if mask is not None:
            token_masks = {k: v for k, v in mask.items() if k != "input"}
        if mask is not None and "input" in mask:
            x_ctx = self._mask_input(x, mask["input"].to(x.device))
            # _action_from_mask already embeds the triple via ActionEmbed.
            act = self._action_from_mask(mask["input"].to(x.device)) \
                if action is None else action
        else:
            x_ctx, act = x, action
        z_ctx = self.encode(x_ctx)
        z_tgt = self.encode(x)  # same weights, second pass, gradients flow
        z_pred = self.predict(z_ctx, act)
        return {
            "context": z_ctx,
            "latents": z_tgt,
            "targets": z_tgt,
            "predicted": z_pred,
            "mask": token_masks or None,
            "action": act,
        }

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load weights, transparently remapping pre-rebuild encoder key names."""
        from models.encoder import remap_legacy_encoder_keys

        return super().load_state_dict(
            remap_legacy_encoder_keys(state_dict), strict=strict, assign=assign)

    def update_ema(self) -> None:
        """No teacher: no-op satisfying the Trainer contract."""

    def update_running_stats(self, latents: dict) -> None:
        """No running statistics on the bare trunk."""

    def latent_variance(self, latents: dict) -> float:
        """Collapse diagnostic: mean per-dim variance across levels."""
        vals = [latents[n].transpose(1, 2).reshape(-1, latents[n].size(1)).var(dim=0).mean()
                for n in latents]
        return torch.stack(vals).mean().item()

    # -- scoring (fp32, eval mode) ----------------------------------------
    @torch.no_grad()
    def score(self, x: torch.Tensor, n_probes: int = 4) -> dict:
        """Probe-query anomaly evidence fused to (B, W).

        Each probe masks one contiguous quarter of the window, encodes the
        masked view, predicts the full-view latents conditioned on the probe
        descriptor, and scores |pred - target| on masked tokens only. Probe
        maps are mean-fused so every timestep is queried equally.
        """
        was_training = self.training
        self.eval()
        b, _, window = x.shape
        z_tgt = self.encode(x)
        probe_maps: dict[str, list] = {n: [] for n in self.level_names}
        for i in range(max(int(n_probes), 1)):
            lo, hi = (i * window) // max(int(n_probes), 1), \
                     ((i + 1) * window) // max(int(n_probes), 1)
            m = torch.zeros((b, window), device=x.device, dtype=torch.bool)
            m[:, lo:hi] = True
            z_ctx = self.encode(self._mask_input(x, m))
            pred = self.predict(z_ctx, self._action_from_mask(m))
            for idx, name in enumerate(self.level_names):
                stride = self.level_strides[idx]
                diff = (pred[name][:, 0] - z_tgt[name]).abs().mean(dim=1)
                t = z_tgt[name].size(-1)
                tm = m.float()
                if stride > 1:
                    tm = F.max_pool1d(tm.unsqueeze(1), kernel_size=stride,
                                      stride=stride).squeeze(1)
                tm = tm[:, :t]
                steps = diff.repeat_interleave(stride, dim=1)[:, :window]
                cover = tm.repeat_interleave(stride, dim=1)[:, :window]
                probe_maps[name].append((steps, cover))
        sums = x.new_zeros((b, window), dtype=torch.float32)
        level_maps: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self.level_names):
            acc = x.new_zeros((b, window), dtype=torch.float32)
            cnt = x.new_zeros((b, window), dtype=torch.float32)
            for steps, cover in probe_maps[name]:
                acc = acc + steps.float()
                cnt = cnt + cover.float()
            level = acc / cnt.clamp(min=1.0)
            level_maps[name] = level
            sums = sums + level
        fused = sums / max(len(self.level_names), 1)
        if was_training:
            self.train()
        return {"fused": fused, "levels": level_maps, "signals": {}}
