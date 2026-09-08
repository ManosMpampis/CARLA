"""True-LeWM model: two-stream input masking over ONE shared encoder.

Stream 1 (context): X -> mask_input(X) -> Encoder -> z_ctx
                    -> Predictor(z_ctx, action) -> z_pred
Stream 2 (target):  X -> Encoder -> z_tgt   (same weights, second pass,
                    NO detach: end-to-end LeWM training, collapse is handled
                    by SIGReg in the criterion, not by stop-gradient/EMA).

Encoder: any pyramid encoder exposing level_names / level_dims /
level_strides (PyramidEncoder or FourierPyramidEncoder). Predictor per
level (T1): "adaln" (per-layer AdaLN transformer, LeWM-style, default) or
"cond" (CondPredictor wrapper legacy arm). Masking already happened in the
INPUT space, so predictors run with mask_pos=None (latent mask tokens
dormant). Optional disposable projector (T2): SIGReg sees projections,
scoring never does.

Action semantics (T3): per-window [mask_ratio, mask_center, mask_span,
band_id] quad embedded by ActionEmbed (band 0 = time probe, 1..B =
frequency-band probes for T6 scoring).
"""
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conditioner import ActionEmbed, Projector
from models.predictor import (AdaLNTransformerPredictor, CondPredictor,
                              MaskedReconPredictor)


def _notch_band(x: torch.Tensor, band: int, n_bands: int) -> torch.Tensor:
    """Zero one frequency third of the input spectrum (T6 band probe)."""
    xf = torch.fft.rfft(x, dim=-1)
    n_freq = xf.size(-1)
    lo, hi = (band * n_freq) // n_bands, ((band + 1) * n_freq) // n_bands
    mask = torch.ones(n_freq, device=x.device, dtype=torch.bool)
    mask[lo:hi] = False
    return torch.fft.irfft(xf * mask, n=x.size(-1), dim=-1)


class TrueLeWMModel(nn.Module):
    """Shared-encoder two-stream world model for masked latent prediction."""

    def __init__(self, encoder: nn.Module, action_dim: int = 16,
                 predictor_kwargs: dict | None = None,
                 use_projector: bool = False, projector_hidden=None,
                 time_probes: int = 4, band_probes: int = 0, n_bands: int = 3):
        super().__init__()
        self.encoder = encoder
        names = cast("list[str]", getattr(encoder, "level_names"))
        dims = cast("list[int]", getattr(encoder, "level_dims"))
        strides = cast("list[int]", getattr(encoder, "level_strides", [1] * len(names)))
        self.level_names, self.level_dims = list(names), [int(d) for d in dims]
        self.level_strides = [int(s) for s in strides]

        pk = dict(predictor_kwargs or {})
        arch = pk.pop("arch", "adaln")
        self.predictors = nn.ModuleDict({
            n: self._build_predictor(arch, d, int(action_dim), pk)
            for n, d in zip(self.level_names, self.level_dims)})
        self.action = ActionEmbed(action_dim=int(action_dim), in_dim=4)
        self.action_dim = int(action_dim)

        self.projectors: nn.ModuleDict | None = None
        if use_projector:
            self.projectors = nn.ModuleDict({
                n: Projector(d, hidden=projector_hidden)
                for n, d in zip(self.level_names, self.level_dims)})

        self.time_probes = int(time_probes)
        self.band_probes = int(band_probes)
        self.n_bands = int(n_bands)

        # Trainer contract handles (None unless an EMA/codebook arm).
        self.target_encoder = None
        self.codebook = None
        self.anti_collapse = "sigreg"
        self.encoder_frozen = False

    @staticmethod
    def _build_predictor(arch: str, dim: int, action_dim: int, pk: dict):
        if arch == "adaln":
            return AdaLNTransformerPredictor(dim, action_dim=action_dim, **pk)
        if arch == "cond":
            return CondPredictor(MaskedReconPredictor(dim, **pk), dim,
                                 action_dim=action_dim)
        raise ValueError(f"Invalid true-LeWM predictor arch {arch}")

    # -- masking helpers -------------------------------------------------
    @staticmethod
    def _mask_input(x: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """Zero masked input steps (post-standardization zeros = channel mean)."""
        m = input_mask.to(torch.bool)
        while m.ndim < x.ndim:
            m = m.unsqueeze(1)
        return x.masked_fill(m.expand_as(x), 0.0)

    def _action_from_mask(self, input_mask: torch.Tensor,
                          band: int = 0) -> torch.Tensor:
        """Describe the probe as a (B, 4) quad for the ActionEmbed."""
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
        band_id = torch.full_like(ratio, float(band) / max(self.n_bands, 1))
        return self.action(torch.stack([ratio, center, span, band_id], dim=1))

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
            # _action_from_mask already embeds the quad via ActionEmbed.
            act = self._action_from_mask(mask["input"].to(x.device)) \
                if action is None else action
        else:
            x_ctx, act = x, action
        z_ctx = self.encode(x_ctx)
        z_tgt = self.encode(x)  # same weights, second pass, gradients flow
        z_pred = self.predict(z_ctx, act)
        out: dict[str, object] = {
            "context": z_ctx,
            "latents": z_tgt,
            "targets": z_tgt,
            "predicted": z_pred,
            "mask": token_masks or None,
            "action": act,
        }
        if self.projectors is not None:
            pooled = {n: torch.cat([z_ctx[n], z_tgt[n]], dim=-1)
                      for n in z_ctx}
            out["projected"] = {
                n: self.projectors[n](
                    pooled[n].transpose(1, 2).reshape(-1, pooled[n].size(1)))
                for n in pooled}
        return out

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
    def score(self, x: torch.Tensor) -> dict:
        """Probe-query anomaly evidence fused to (B, W).

        Time probes (band 0): contiguous input blocks, error on masked
        tokens only. Band probes (T6): one spectrum third notched per probe
        over the full window, dense error, reported as per-band signal
        channels for the calibrator fusion. Scoring never touches the
        disposable projectors.
        """
        was_training = self.training
        self.eval()
        b, _, window = x.shape
        z_tgt = self.encode(x)
        level_acc: dict[str, list] = {n: [] for n in self.level_names}
        band_maps: dict[str, list] = {}
        for i in range(max(self.time_probes, 1)):
            lo, hi = (i * window) // max(self.time_probes, 1), \
                     ((i + 1) * window) // max(self.time_probes, 1)
            m = torch.zeros((b, window), device=x.device, dtype=torch.bool)
            m[:, lo:hi] = True
            self._accumulate_probe(x, m, 0, z_tgt, level_acc, None)
        for band in range(max(self.band_probes, 0)):
            xb = _notch_band(x.float(), band, self.n_bands)
            acc: list = []
            self._accumulate_probe(xb, None, band + 1, z_tgt, level_acc, acc)
            band_maps[f"band_{band}"] = acc
        sums = x.new_zeros((b, window), dtype=torch.float32)
        level_maps: dict[str, torch.Tensor] = {}
        for name in self.level_names:
            acc = x.new_zeros((b, window), dtype=torch.float32)
            cnt = x.new_zeros((b, window), dtype=torch.float32)
            for steps, cover in level_acc[name]:
                acc = acc + steps.float()
                cnt = cnt + cover.float()
            level = acc / cnt.clamp(min=1.0)
            level_maps[name] = level
            sums = sums + level
        fused = sums / max(len(self.level_names), 1)
        signals: dict[str, torch.Tensor] = {}
        for band, acc in band_maps.items():
            stacked = torch.stack([s.float() for s in acc], dim=0).mean(dim=0)
            signals[band] = stacked
        if was_training:
            self.train()
        return {"fused": fused, "levels": level_maps, "signals": signals}

    @torch.no_grad()
    def _accumulate_probe(self, x_view: torch.Tensor,
                          m: torch.Tensor | None, band: int, z_tgt: dict,
                          level_acc: dict, band_acc: list | None) -> None:
        """Encode one probe view, predict with its descriptor, file errors.

        Time probes pass the boolean input mask (error on masked tokens
        only, filed into level_acc as (steps, cover)). Band probes pass
        m=None with a pre-notched view (dense error, filed into band_acc).
        """
        b, _, window = x_view.shape
        if m is None:
            z_ctx = self.encode(x_view)
            act = self._band_action(b, window, band, x_view.device)
            dense = True
        else:
            z_ctx = self.encode(self._mask_input(x_view, m))
            act = self._action_from_mask(m, band)
            dense = False
        pred = self.predict(z_ctx, act)
        for idx, name in enumerate(self.level_names):
            stride = self.level_strides[idx]
            diff = (pred[name][:, 0] - z_tgt[name]).abs().mean(dim=1)
            t = z_tgt[name].size(-1)
            if dense:
                tm = torch.ones((b, t), device=diff.device)
            else:
                tm = m.float()
                if stride > 1:
                    tm = F.max_pool1d(tm.unsqueeze(1), kernel_size=stride,
                                      stride=stride).squeeze(1)
                tm = tm[:, :t]
            steps = diff.repeat_interleave(stride, dim=1)[:, :window]
            if band_acc is not None:
                band_acc.append(steps)
            else:
                cover = tm.repeat_interleave(stride, dim=1)[:, :window]
                level_acc[name].append((steps, cover))

    def _band_action(self, b: int, window: int, band: int, device) -> torch.Tensor:
        """Neutral-geometry descriptor carrying only the band id."""
        quad = torch.zeros((b, 4), device=device)
        quad[:, 0], quad[:, 1], quad[:, 2] = 1.0, 0.5, 1.0
        quad[:, 3] = float(band) / max(self.n_bands, 1)
        return self.action(quad)
