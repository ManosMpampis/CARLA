"""Shared convolutional building blocks (configurable, documented).

All blocks are plain classes with methods. Kernel/stride/depth choices
come from config; scalar args broadcast over the block's convolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedEngine(nn.Module):
    def __init__(self, mask_type):
        super().__init__()
        self.mask_type = mask_type
        if self.mask_type not in ("injection", "zero"):
            raise ValueError("mask_type must be injection or zero")
        if self.mask_type == "injection":
            # Add subanomaly to input to simulate a mask token and pass it to the device.
            self.mask_function = self._inject_mask_input
        elif self.mask_type == "zero":
            # Zero out parts of the input and pass it to the device.
            self.mask_function = self._zero_mask_input
            
    @staticmethod
    def _zero_mask_input(x, input_mask):
        m = input_mask.to(torch.bool)
        while m.ndim < x.ndim:
            m = m.unsqueeze(1)
        return x.masked_fill(m.expand_as(x), 0.0)
    
    def forward(self, x):
        """
        """

        x_inj, m = self.mask_function(x)
        m_bool = m.to(torch.bool) if m.dtype == torch.bool else (m > 0.5)
        
        return {"current_state": x,
                "future_state": x_inj,
                "query_start": 0,
                "target_range": (0, x.size(-1)),
                "mask": m_bool,
                "mask_float": m}
    
class ForecastEngine(nn.Module):
    def __init__(self, encoder_strides, input_window, context_crop=None, target_crop=None):
        super().__init__()
        self.context_crop = self._interval(context_crop)
        self.target_crop = self._interval(target_crop)
        self.input_window = input_window
        if (self.context_crop is None) != (self.target_crop is None):
            raise ValueError("set both context_crop and target_crop, or neither")
        self.encoder_strides = encoder_strides
        if input_window % self.encoder_strides:
            raise ValueError("window must be divisible by feature extractor stride")
        
        return

    def forward(self, x):
        context, target = self.resolve_crop_ranges(x.size(-1))
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.encoder_frozen):
            if self.feature_extraction == "full":
                z = self.encoder(x)
                return z[..., context[0]:context[1]], z[..., target[0]:target[1]]
            s = self.total_stride
            return {"current_state": x[..., context[0]*s:context[1]*s],
                    "future_state": x[..., target[0]*s:target[1]*s],
                    "query_start": self._query_start(),
                    "target_range": self.target_input_range(x.size(-1))}

    def _query_start(self):
        return self.target_crop[0] - self.context_crop[1]
    
    def target_input_range(self, window):
        _, (a, b) = self.resolve_crop_ranges(window)
        return a * self.total_stride, b * self.total_stride
        
    def resolve_crop_ranges(self, window):
        """Validate crop geometry without constructing an encoder or model."""
        tokens = window // self.encoder_strides
        if self.context_crop is None:
            if tokens < 2 or tokens % 2:
                raise ValueError("default half split requires an even feature length >= 2")
            context, target = (0, tokens // 2), (tokens // 2, tokens)
        else:
            context = (self.context_crop[0] // self.encoder_strides, self.context_crop[1] // self.encoder_strides)
            target = (self.target_crop[0] // self.encoder_strides, self.target_crop[1] // self.encoder_strides)
        if not context[1] <= target[0] < target[1] <= tokens:
            raise ValueError(
                f"context_crop={self.context_crop} and target_crop={self.target_crop} must precede "
                f"one another without overlap, inside window={window} "
                f"(feature stride={self.encoder_strides}, tokens={tokens})")
        return context, target

    @staticmethod
    def _interval(crop):
        if crop is None:
            return None
        if (not isinstance(crop, (list, tuple)) or len(crop) != 2
                or any(type(v) is not int for v in crop)
                or not 0 <= crop[0] < crop[1]):
            raise ValueError("crops must be integer [start, end] with 0 <= start < end")
        return tuple(crop)
    