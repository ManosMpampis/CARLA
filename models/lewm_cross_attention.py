"""Frozen LEWM features -> convolutional Q/K/V -> attention -> mirrored head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convolutions import _init_weights
from models.recon_head import build_mirrored_head
from utils.reconstruction_scores import normalize_score_mode, reconstruction_score_map


class ConvQKVEncoder(nn.Module):
    """Three length-preserving convolutions, with independent Q/K and V widths."""

    def __init__(self, in_channels, qk_channels=8, value_channels=8,
                 kernel_size=3):
        super().__init__()
        if min(in_channels, qk_channels, value_channels, kernel_size) < 1:
            raise ValueError("Q/K/V channels and kernel_size must be positive")
        if kernel_size % 2 != 1:
            raise ValueError("Q/K/V kernel_size must be odd to preserve length")
        self.query = nn.Conv1d(in_channels, qk_channels, kernel_size,
                               padding=kernel_size // 2)
        self.key = nn.Conv1d(in_channels, qk_channels, kernel_size,
                             padding=kernel_size // 2)
        self.value = nn.Conv1d(in_channels, value_channels, kernel_size,
                               padding=kernel_size // 2)
        self.apply(_init_weights)

    def forward(self, context, query_input):
        return {"key": self.key(context), "value": self.value(context),
                "query": self.query(query_input)}


class CrossAttentionPredictor(nn.Module):
    """Scaled cross attention and reconstruction, without a Query residual bypass."""

    def __init__(self, head, qk_channels=8, value_channels=8, num_heads=1,
                 dropout=0.0, input_projection=None, position_encoding="none",
                 position_on="both", position_scale=1.0):
        super().__init__()
        if num_heads < 1 or qk_channels % num_heads or value_channels % num_heads:
            raise ValueError("num_heads must divide both qk_channels and value_channels")
        if not 0 <= dropout < 1:
            raise ValueError("attention dropout must be in [0, 1)")
        if position_encoding not in ("none", "sinusoidal"):
            raise ValueError("position_encoding must be none or sinusoidal")
        if position_on not in ("query", "key", "both"):
            raise ValueError("position_on must be query, key, or both")
        if (isinstance(position_scale, bool) or not isinstance(position_scale, (int, float))
                or not math.isfinite(position_scale) or position_scale < 0):
            raise ValueError("position_scale must be finite and nonnegative")
        self.head = head
        self.input_projection = input_projection if input_projection is not None else nn.Identity()
        self.num_heads = num_heads
        self.dropout = dropout
        self.position_encoding = position_encoding
        self.position_on = position_on
        self.position_scale = float(position_scale)

    def _with_position(self, z, start):
        """Add fixed sinusoids in feature-token units, supporting any length/width."""
        channels, length = z.shape[1:]
        positions = torch.arange(start, start + length, device=z.device, dtype=torch.float32)
        frequencies = torch.exp(torch.arange(0, channels, 2, device=z.device,
                                            dtype=torch.float32) * (-math.log(10000.0) / channels))
        angles = frequencies[:, None] * positions[None, :]
        encoding = torch.empty(channels, length, device=z.device, dtype=torch.float32)
        encoding[0::2] = angles.sin()
        encoding[1::2] = angles[:channels // 2].cos()
        return z + self.position_scale * encoding.to(z.dtype).unsqueeze(0)

    def forward(self, qkv, *, query_start=0):
        # Enrich attention addressing only. Keep content Q/K/V unchanged for
        # SIGReg and the clean Query reconstruction target.
        query, key = qkv["query"], qkv["key"]
        if self.position_encoding != "none" and self.position_scale:
            if self.position_on in ("query", "both"):
                query = self._with_position(query, query_start)
            if self.position_on in ("key", "both"):
                key = self._with_position(key, -key.size(-1))

        def heads(z):
            b, c, t = z.shape
            return z.reshape(b, self.num_heads, c // self.num_heads, t).transpose(-1, -2)

        attended = F.scaled_dot_product_attention(
            heads(query), heads(key), heads(qkv["value"]),
            dropout_p=self.dropout if self.training else 0.0)
        b, _, t, _ = attended.shape
        attended = attended.transpose(-1, -2).reshape(b, -1, t)
        return self.head(self.input_projection(attended))


class CrossAttentionLeWM(nn.Module):
    """Two crop LEWM with a fixed feature extractor and trainable Q/K/V encoder.

    Crop intervals are half-open FEATURE TOKEN indices. With stride S,
    [a,b] refers to raw input [a*S,b*S]. Omitted intervals split in half.
    ``observed`` obtains Query from the second crop (conditional reconstruction).
    ``context`` obtains Query from the first crop (past-only forecasting).
    Separate extraction isolates receptive fields; full extraction is available
    for the observed arm to literally crop one full-window feature map.
    """

    def __init__(self, encoder, freeze_encoder=True, target="features", query_source="observed",
                 context_crop=None, target_crop=None, feature_extraction="separate",
                 qk_channels=8, value_channels=8, kernel_size=3, num_heads=1,
                 attention_dropout=0.0, head_channels=None, norm="batch",
                 dropout=0.0, score_mode="l1", position_encoding="none",
                 position_on="both", position_scale=1.0):
        super().__init__()
        if target not in ("features", "input", "query"):
            raise ValueError("target must be features, input, or query")
        if query_source not in ("observed", "context"):
            raise ValueError("query_source must be observed or context")
        if feature_extraction not in ("separate", "full"):
            raise ValueError("feature_extraction must be separate or full")
        if query_source == "context" and feature_extraction != "separate":
            raise ValueError("forecasting requires separate feature extraction to prevent future leakage")
        if (context_crop is None) != (target_crop is None):
            raise ValueError("set both context_crop and target_crop, or neither")
        self.context_crop = self._interval(context_crop)
        self.target_crop = self._interval(target_crop)
        self.encoder = encoder
        if not isinstance(freeze_encoder, bool):
            raise ValueError("freeze_encoder must be true or false")
        self.encoder_frozen = freeze_encoder
        self.encoder.requires_grad_(not freeze_encoder)
        if freeze_encoder:
            self.encoder.eval()
        self.target = target
        self.query_source = query_source
        self.feature_extraction = feature_extraction
        self.total_stride = int(encoder.total_stride)
        self.qkv_encoder = ConvQKVEncoder(encoder.output_dims, qk_channels,
                                         value_channels, kernel_size)
        out_channels = {"features": encoder.output_dims,
                        "input": encoder.channel_ladder[0], "query": qk_channels}[target]
        # Accept old configs only when they agree with the automatic mirror.
        # The encoder's deepest width must never become the attention width.
        if head_channels is not None and list(head_channels) != list(encoder.channel_ladder[1:-1]):
            raise ValueError("head_channels is derived from the encoder; remove the override")
        head = build_mirrored_head(encoder, norm=norm, dropout=dropout,
                                   out_channels=out_channels, upsample=target == "input")
        input_projection = nn.Conv1d(value_channels, encoder.output_dims, kernel_size=1) \
            if value_channels != encoder.output_dims else nn.Identity()
        input_projection.apply(_init_weights)
        self.predictor = CrossAttentionPredictor(head, qk_channels, value_channels,
                                                 num_heads, attention_dropout,
                                                 input_projection=input_projection,
                                                 position_encoding=position_encoding,
                                                 position_on=position_on,
                                                 position_scale=position_scale)
        self.score_mode = normalize_score_mode(score_mode)
        self.level_names = ["L0"]
        self.level_dims = [value_channels]
        self.level_strides = [self.total_stride]
        self.target_encoder = None
        self.codebook = None
        self.anti_collapse = "sigreg"
        # Persist task semantics as well as weights: equal-shaped arms must
        # not silently resume/score with a different Query source or crop.
        self._task = dict(target=target, query_source=query_source,
                          feature_extraction=feature_extraction,
                          context_crop=self.context_crop, target_crop=self.target_crop,
                          qk_channels=qk_channels, value_channels=value_channels,
                          kernel_size=kernel_size, num_heads=num_heads,
                          attention_dropout=attention_dropout,
                          position_encoding=position_encoding,
                          position_on=position_on, position_scale=float(position_scale),
                          reconstructor="encoder_mirror_v2",
                          freeze_encoder=freeze_encoder, norm=norm, dropout=dropout,
                          total_stride=self.total_stride,
                          enc_strides=list(encoder.enc_strides),
                          channel_ladder=list(encoder.channel_ladder))

    @staticmethod
    def _interval(crop):
        if crop is None:
            return None
        if (not isinstance(crop, (list, tuple)) or len(crop) != 2
                or any(type(v) is not int for v in crop)
                or not 0 <= crop[0] < crop[1]):
            raise ValueError("crops must be integer [start, end] with 0 <= start < end")
        return tuple(crop)

    def get_extra_state(self):
        return self._task

    def set_extra_state(self, state):
        # Checkpoints predating positional attention retain their original
        # no-position behavior; enabled/changed positional tasks cannot resume them.
        state = {"position_encoding": "none", "position_on": "both",
                 "position_scale": 1.0, **state}
        if state != self._task:
            raise ValueError(f"checkpoint task differs from configured task: {state} != {self._task}")

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def crop_ranges(self, window):
        if window % self.total_stride:
            raise ValueError("window must be divisible by feature extractor stride")
        tokens = window // self.total_stride
        if self.context_crop is None:
            if tokens < 2 or tokens % 2:
                raise ValueError("default half split requires an even feature length >= 2")
            context, target = (0, tokens // 2), (tokens // 2, tokens)
        else:
            context, target = (self.context_crop[0] // self.total_stride, self.context_crop[1] // self.total_stride), (self.target_crop[0] // self.total_stride, self.target_crop[1] // self.total_stride)
        if not context[1] <= target[0] < target[1] <= tokens:
            raise ValueError("context must precede target without overlap, inside the feature map")
        return context, target

    def target_input_range(self, window):
        _, (a, b) = self.crop_ranges(window)
        return a * self.total_stride, b * self.total_stride

    def _features(self, x):
        context, target = self.crop_ranges(x.size(-1))
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.encoder_frozen):
            if self.feature_extraction == "full":
                z = self.encoder(x)
                return z[..., context[0]:context[1]], z[..., target[0]:target[1]]
            s = self.total_stride
            return (self.encoder(x[..., context[0]*s:context[1]*s]),
                    self.encoder(x[..., target[0]*s:target[1]*s]))

    def _qkv(self, context, second):
        query_input = second if self.query_source == "observed" else F.interpolate(
            context, size=second.size(-1), mode="linear", align_corners=False)
        return self.qkv_encoder(context, query_input)

    def _query_start(self):
        """Target offset from the context's exclusive end, in feature tokens."""
        return 0 if self.target_crop is None else self.target_crop[0] - self.context_crop[1]

    def forward(self, x, mask=None, action=None):
        if action is not None:
            raise ValueError("use mask with input or X_inj for injected LEWM training")
        first, second = self._features(x)
        clean_qkv = self.qkv_encoder(first, second)
        if mask is not None:
            if "X_inj" in mask:
                injected = mask["X_inj"].to(device=x.device, dtype=x.dtype)
                if injected.shape != x.shape:
                    raise ValueError("X_inj must have the same shape as input")
            else:
                injected = x.masked_fill(mask["input"].to(x.device).bool().unsqueeze(1), 0)
            inj_first, inj_second = self._features(injected)
        else:
            inj_first, inj_second = first, second
        qkv = self._qkv(inj_first, inj_second)
        prediction = self.predictor(qkv, query_start=self._query_start())
        a, b = self.target_input_range(x.size(-1))
        target = {"features": second, "input": x[..., a:b],
                  "query": clean_qkv["query"]}[self.target]
        return {"recon": prediction, "target": target,
                "qkv": qkv, "clean_qkv": clean_qkv,
                "context": {"L0": qkv["value"]},
                "latents": {"L0": clean_qkv["value"]},
                "targets": {"L0": target}, "predicted": {"L0": prediction}}

    @torch.no_grad()
    def predict_next(self, context, target_tokens=None):
        """Forecast from ONLY the raw context crop; output uses this arm's domain."""
        if self.query_source != "context":
            raise ValueError("predict_next requires query_source=context; observed Query needs the next crop")
        was_training = self.training
        self.eval()
        try:
            if context.size(-1) % self.total_stride:
                raise ValueError("context length must be divisible by feature stride")
            first = self.encoder(context.float())
            if target_tokens is None:
                if self.target_crop is None:
                    target_tokens = first.size(-1)
                else:
                    target_tokens = self.target_crop[1] - self.target_crop[0]
            if type(target_tokens) is not int or target_tokens < 1:
                raise ValueError("target_tokens must be a positive integer")
            query_input = F.interpolate(first, size=target_tokens, mode="linear",
                                        align_corners=False)
            return self.predictor(self.qkv_encoder(first, query_input),
                                  query_start=self._query_start())
        finally:
            self.train(was_training)

    @torch.no_grad()
    def score(self, x):
        """Return only the target crop's score map, always in input timesteps."""
        was_training = self.training
        self.eval()
        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                out = self(x.float())
                fused, _ = reconstruction_score_map(out["recon"], out["target"], self.score_mode)
                if self.target != "input":
                    fused = fused.repeat_interleave(self.total_stride, dim=-1)
                return {"fused": fused, "levels": {"L0": fused}, "signals": {}}
        finally:
            self.train(was_training)

    def update_ema(self):
        """No teacher in LEWM."""

    def update_running_stats(self, latents):
        """No additional running statistics."""

    def latent_variance(self, latents):
        z = latents["L0"]
        return z.transpose(1, 2).reshape(-1, z.size(1)).var(dim=0, unbiased=False).mean().item()
