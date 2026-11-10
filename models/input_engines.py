"""Prepare raw current and target views before feature extraction."""

from dataclasses import dataclass

import torch
from torch import nn

from utils.masking_steered import SubAnomalyMaskCollator


@dataclass(frozen=True)
class InputViews:
    current_input: torch.Tensor
    target_input: torch.Tensor
    current_range: tuple[int, int]
    target_range: tuple[int, int]
    mask_target: torch.Tensor | None = None


class MaskedEngine(nn.Module):
    def __init__(self, mask_type="injection", *, sample=True, min_ratio=0.1, max_ratio=0.9):
        super().__init__()
        if mask_type not in ("injection", "zero"):
            raise ValueError("mask_type must be injection or zero")
        if not 0 < min_ratio <= max_ratio <= 1:
            raise ValueError("mask ratios must satisfy 0 < min_ratio <= max_ratio <= 1")
        self.mask_type = mask_type
        self.sample = sample
        self.collator = SubAnomalyMaskCollator(min_ratio=min_ratio, max_ratio=max_ratio)

    def forward(self, x, mask=None, *, corrupt=True):
        batch, _, window = x.shape
        if mask is None and corrupt and self.sample:
            source = x if self.mask_type == "injection" else None
            mask = self.collator(batch, window, [1], source)
        if mask is None:
            return InputViews(x, x, (0, window), (0, window))
        if not isinstance(mask, dict) or "input" not in mask:
            raise ValueError("mask must provide an 'input' tensor of shape (B, W)")
        known = mask["input"].to(device=x.device, dtype=torch.bool)
        if known.shape != (batch, window):
            raise ValueError(f"mask input shape {tuple(known.shape)} must equal {(batch, window)}")
        if "X_inj" in mask:
            current = mask["X_inj"].to(device=x.device, dtype=x.dtype)
            if current.shape != x.shape:
                raise ValueError("X_inj must have the same shape as input")
        else:
            current = x.masked_fill(known.unsqueeze(1), 0.0)
        return InputViews(current, x, (0, window), (0, window), known)

    @staticmethod
    def target_input_range(window):
        return 0, window


class ForecastEngine(nn.Module):
    def __init__(self, encoder_strides=1, input_window=None, context_crop=None, target_crop=None):
        super().__init__()
        if type(encoder_strides) is not int or encoder_strides < 1:
            raise ValueError("encoder_strides must be a positive integer")
        self.encoder_strides = encoder_strides
        self.context_crop = self._interval(context_crop)
        self.target_crop = self._interval(target_crop)
        if (self.context_crop is None) != (self.target_crop is None):
            raise ValueError("set both context_crop and target_crop, or neither")
        if input_window is not None:
            self.resolve_crop_ranges(input_window)

    @staticmethod
    def _interval(crop):
        if crop is None:
            return None
        if (not isinstance(crop, (tuple, list)) or len(crop) != 2
                or any(type(v) is not int for v in crop) or not 0 <= crop[0] < crop[1]):
            raise ValueError("crops must be integer [start, end] with 0 <= start < end")
        return tuple(crop)

    def resolve_crop_ranges(self, window):
        stride = self.encoder_strides
        if window < 2 * stride or window % stride:
            raise ValueError("window must contain at least two tokens and be divisible by feature stride")
        context, target = self.context_crop, self.target_crop
        if context is None:
            if window % (2 * stride):
                raise ValueError("default half split requires an even feature length")
            context, target = (0, window // 2), (window // 2, window)
        if (any(edge % stride for edge in (*context, *target))
                or not 0 <= context[0] < context[1] <= target[0] < target[1] <= window):
            raise ValueError("crops must be nonempty, ordered and stride-aligned inside the window")
        return context, target

    def target_input_range(self, window):
        return self.resolve_crop_ranges(window)[1]

    def forward(self, x, mask=None, *, corrupt=True):
        if mask is not None:
            raise ValueError("forecast views do not accept a corruption mask")
        context, target = self.resolve_crop_ranges(x.size(-1))
        return InputViews(x[..., context[0]:context[1]], x[..., target[0]:target[1]],
                          context, target)
