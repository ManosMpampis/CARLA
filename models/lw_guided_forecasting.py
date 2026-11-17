"""Frozen LEWM features -> convolutional Q/K/V -> attention -> mirrored head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import Conv1dSamePadding, StackedConvBlock, CrossAttention, ConvBlock1d
from utils.reconstruction_scores import normalize_score_mode, reconstruction_score_map

class ForecastingEngine(nn.Module):
    def __init__(self, history_ratio, future_ratio):
        super().__init__()
        self.hr = history_ratio if history_ratio is not None else 1
        self.fr = future_ratio  if future_ratio is not None else 1
        self.total_part = self.hr + self.fr

    def forward(self, x):
        if x.size(-1) % self.total_part:
            raise ValueError("window must be divisible by feature extractor stride")
        tokens = x // self.total_stride

        context, target = (0, tokens * self.hr), (tokens * self.hr, tokens * self.fr)
        if not context[1] <= target[0] < target[1] <= tokens:
            raise ValueError("context must precede target without overlap, inside the feature map")
        history = x[..., context[0]:context[1]]
        future = x[..., target[0]:target[1]]
        return history, future

class CrossAttentionLeWM(nn.Module):
    """Two crop LEWM with a fixed feature extractor and trainable Q/K/V encoder.

    Crop intervals are half-open FEATURE TOKEN indices. With stride S,
    [a,b] refers to raw input [a*S,b*S]. Omitted intervals split in half.
    ``observed`` obtains Query from the second crop (conditional reconstruction).
    Separate extraction isolates receptive fields; full extraction is available
    for the observed arm to literally crop one full-window feature map.
    """

    def __init__(self, encoder: nn.Module, freeze_encoder=True, target="features", context_crop=None, target_crop=None,
                 qkv: dict = {"channels": [8], "kernels": [3], "strides": [1], "dilation": [1], "act": "gelu", "norm": "barch", "dropout": 0.1, "causal": True}, 
                 cross_attentnion: dict = {"num_heads":1, "position_encoding":"none", "position_on":"both", "position_scale":1.0, "attention_dropout":0.0},
                 head: dict = {"channels": [64, 32], "kernels": [5, 5], "strides": [1], "dilation": [1], "act": "gelu", "norm": "barch", "dropout": 0.1, "causal": True},
                 score_mode="l1", amp={"enabled": False, "dtype": torch.bfloat16}):
        super().__init__()
        if target not in ("features", "input", "query"):
            raise ValueError("target must be features, query, or input")
        if (context_crop is None) != (target_crop is None):
            raise ValueError("set both context_crop and target_crop, or neither")
        
        self.context_ratio = self._interval(context_crop)
        self.target_ratio = self._interval(target_crop)
        self.engine = ForecastingEngine(self.context_ratio, self.target_ratio)

        self.encoder = encoder
        if not isinstance(freeze_encoder, bool):
            raise ValueError("freeze_encoder must be true or false")
        self.encoder_frozen = freeze_encoder
        self.encoder.requires_grad_(not freeze_encoder)
        if freeze_encoder:
            self.encoder.eval()
        
        self.target = target

        self.query = StackedConvBlock(encoder.output_dims, qkv["channels"], kernels=qkv["kernels"], stride=qkv["stride"], dilation=qkv["dilation"], act=qkv["act"], norm=qkv["norm"], dropout=qkv["dropout"], causal=qkv["causal"]) #Trainable actions from the future
        self.key = StackedConvBlock(encoder.output_dims, qkv["channels"], kernels=qkv["kernels"], stride=qkv["stride"], dilation=qkv["dilation"], act=qkv["act"], norm=qkv["norm"], dropout=qkv["dropout"], causal=qkv["causal"])
        self.value = StackedConvBlock(encoder.output_dims, qkv["channels"], kernels=qkv["kernels"], stride=qkv["stride"], dilation=qkv["dilation"], act=qkv["act"], norm=qkv["norm"], dropout=qkv["dropout"], causal=qkv["causal"])

        self.out_channels = {"input": encoder.channel_ladder[0], "features": encoder.output_dims, "query": qkv["channels"]}

        self.predictor = nn.Sequential(
            CrossAttention(**cross_attentnion),
            Conv1dSamePadding(qkv["channels"], encoder.output_dims, kernel_size=1) if qkv["channels"] != encoder.output_dims else nn.Identity(),
            StackedConvBlock(encoder.output_dims, qkv["channels"], kernels=head["kernels"], stride=head["stride"], dilation=head["dilation"], act=head["act"], norm=head["norm"], dropout=head["dropout"], causal=head["causal"]),
            ConvBlock1d(qkv["channels"][-1], encoder.channel_ladder[0], encoder.kernels[0], encoder.enc_strides[0], act=head["act"], norm=head["norm"], dropout=head["dropout"], causal=head["caulsal"]) if target == "input" else nn.Identity(),
        )

        self.score_mode = normalize_score_mode(score_mode)
        self.amp = {"enabled":amp["enabled"],
                    "dtype": torch.bfloat16 if (amp["dtype"] == "torch.bfloat16" or amp["dtype"] == torch.bfloat16) else torch.float16}

    @staticmethod
    def _interval(crop):
        if crop is None:
            return None
        if (not isinstance(crop, (list, tuple)) or len(crop) != 2
                or any(type(v) is not int for v in crop)
                or not 0 <= crop[0] < crop[1]):
            raise ValueError("crops must be integer [start, end] with 0 <= start < end")
        return tuple(crop)

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def encode(self, x):
        return self.encoder(x)

    def action(self, x):
        return self.query(x)

    def prediction(self, latent, action):
        k = self.key(latent)
        v = self.value(latent)
        return self.predictor({"key": k, "value": v}, action)

    def fit(self, x):
        history, future = self.engine(x)

        zt = self.encode(history)
        zt_f = self.encode(future)

        # Action
        act = self.action(zt_f)

        # Prediction
        pred = self.prediction(zt, act)
        
        target = {"features": zt_f, "input": future, "query": act}[self.target]
        return {
            "known_latents": zt,
            "unknown_latents": zt_f,
            "predicted": pred,
            "action": act,
            "targets": {"reconstruction": target}
        }
    
    def forward(self, x):
        history, future = self.engine(x)

        zt = self.encode(history)
        zt_f = self.encode(future)

        # Action
        act = self.action(zt_f)

        # Prediction
        pred = self.prediction(zt, act)
        
        target = {"features": zt_f, "input": future, "query": act}[self.target]
        return {
            "known_latents": zt,
            "unknown_latents": zt_f,
            "predicted": pred,
            "action": act,
            "targets": {"reconstruction": target}
        }
    
    @torch.no_grad()
    def score(self, x):
        """Return only the target crop's score map, always in input timesteps."""
        was_training = self.training
        self.eval()
        try:
            with torch.autocast(device_type=x.device.type, enabled=self.amp["enabled"], dtype=self.amp["dtype"]):
                out = self(x)
                score, _ = reconstruction_score_map(out["predicted"], out["unknown_latents"], self.score_mode)
                return score
        finally:
            self.train(was_training)

    def latent_variance(self, latents):
        z = latents
        return z.transpose(1, 2).reshape(-1, z.size(1)).var(dim=0, unbiased=False).mean().item()
