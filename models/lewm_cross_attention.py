"""Frozen LEWM features -> convolutional Q/K/V -> attention -> mirrored head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import _init_weights, CrossAttention, StackedConv1dBlock, ResNetBlock
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

    def __init__(self, encoder, action, kernels=(7,5,3), strides=(1, 1, 1), residual_kernel: int = 5,
                 norm: str = "batch", act: str = "relu", dropout: int = 0.1, attention_heads: int = 1,
                 input_projection=None, position_encoding="sinusoidal", position_on="both", diagonal_mask: bool = False):
        super().__init__()
        encoder_output = encoder.output_dims
        action_output = action.output_dims
        assert encoder_output == action_output, "encoder and action should have the same output dim for cross entropy"
        self.attention_heads = attention_heads
        self.cross_attention = CrossAttention(num_heads=attention_heads, dropout=dropout,
                                            input_projection=input_projection, position_encoding=position_encoding, 
                                            position_on=position_on, diagonal_mask=diagonal_mask)
        kernels = list(kernels)
        if len(kernels) == 0:
            self.pred = nn.Identity()
        else:
            strides = list(strides)
            residual_kernel = list(residual_kernel)
            if len(residual_kernel)==1:
                residual_kernel = residual_kernel * len(strides)

            if residual_kernel[0]>=0:
                self.pred = nn.Sequential(*
                    [
                        ResNetBlock(encoder_output, encoder_output, kernels=kernels, stride=st, residual_kernel=rk, norm=norm, act=act, dropout=dropout) for st, rk in zip(strides, residual_kernel)
                    ]
                )
            else:
                self.pred = StackedConv1dBlock(encoder_output, encoder_output, kernels=kernels, strides=strides, norm=norm, act=act, dropout=dropout)

    def forward(self, k, qv):
        x = self.cross_attention(k, qv)
        x = self.pred(x)
        return x


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
                 position_on="both", position_scale=1.0, diagonal_mask=False):
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
                                                 position_scale=position_scale,
                                                 diagonal_mask=diagonal_mask)
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
                          attention_dropout=attention_dropout, diagonal_mask=diagonal_mask,
                          attention_mask_layout="strict_lower_v1" if diagonal_mask else "none",
                          position_encoding=position_encoding,
                          position_on=position_on, position_scale=float(position_scale),
                          reconstructor="encoder_mirror_v2",
                          freeze_encoder=freeze_encoder, norm=norm, dropout=dropout,
                          total_stride=self.total_stride,
                          enc_strides=list(encoder.enc_strides),
                          channel_ladder=list(encoder.channel_ladder))

    

    def get_extra_state(self):
        return self._task

    def set_extra_state(self, state):
        # Older checkpoints retain unmasked, no-position attention. Changed
        # task settings require a fresh run even when parameter shapes agree.
        # The earlier diagonal-only mask has different task semantics.
        mask_layout = "exclude_diagonal_v1" if state.get("diagonal_mask", False) else "none"
        state = {"attention_mask_layout": mask_layout, "position_encoding": "none", "position_on": "both",
                 "position_scale": 1.0, "diagonal_mask": False, **state}
        if state != self._task:
            raise ValueError(f"checkpoint task differs from configured task: {state} != {self._task}")

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, x):
        
        a = self.action(x) 
        z = self.encoder(a["future_state"])
        zhat = self.encoder(a["current_state"])
        zhat_dot = self.predictor(a["future_state"], a["query_start"])
        return {"prediction": zhat_dot,
                "latents_current_state": z,
                "latents_future_state": zhat,
                "action": a}

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
