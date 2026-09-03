"""Configurable time-series encoders (pyramid + transformer arm).

PyramidEncoder exposes {level: (B, D_l, T_l)} with strides so scorers can
map tokens back to input steps. All per-block structure comes from config;
scalar args broadcast to every block. Defaults reproduce the proven
stem + L1..L3 ladder exactly.
"""
import math

import torch
import torch.nn as nn

from models.blocks import StackedConvBlock


def _broadcast(name, value, n):
    if isinstance(value, (list, tuple)):
        assert len(value) == n, f"{name} must match num blocks"
        return list(value)
    return [value] * n


def remap_legacy_encoder_keys(state_dict: dict) -> dict:
    """Remap pre-rebuild encoder submodule names to the clean layout.

    Old PyramidEncoder used stem/downsample/refine blocks; the rebuilt
    encoder nests them as stem.layers.0 / blocks.i.layers.{0,1} with
    identical shapes, so old checkpoints transfer exactly. New keys pass
    through untouched (idempotent).
    """
    import re

    out = {}
    for key, value in state_dict.items():
        if ".levels." in key:
            key = re.sub(r"\.levels\.(\d+)\.downsample\.", r".blocks.\1.layers.0.", key)
            key = re.sub(r"\.levels\.(\d+)\.refine\.", r".blocks.\1.layers.1.", key)
        if ".stem." in key and ".stem.layers." not in key:
            key = key.replace(".stem.", ".stem.layers.0.")
        out[key] = value
    return out


class PyramidEncoder(nn.Module):
    """Multi-scale convolutional encoder over (B, C, W)."""

    def __init__(self, in_channels: int = 38, stem_channels: int = 32,
                 level_channels=(32, 64, 96), kernel_size=5, strides=(2, 2, 2),
                 depths=2, norm: str = "batch", dropout: bool = True,
                 stem_kernel: int = 7):
        super().__init__()
        n = len(level_channels)
        strides = list(strides)
        assert len(strides) == n
        kernels = _broadcast("kernel_size", kernel_size, n)
        depths = _broadcast("depths", depths, n)
        norms = _broadcast("norm", norm, n)

        # Stem: full-resolution detail (stride 1, kernel 7 by proven default).
        self.stem = StackedConvBlock(in_channels, stem_channels, depth=1,
                                     kernels=[int(stem_kernel)],
                                     stride=1, norm=norms[0], dropout=dropout)
        self.blocks = nn.ModuleList()
        ch_in = stem_channels
        dims = [stem_channels]
        for ch_out, k, s, d, nm in zip(level_channels, kernels, strides, depths, norms):
            kk = list(k) if isinstance(k, (list, tuple)) else [k] * int(d)
            self.blocks.append(StackedConvBlock(ch_in, int(ch_out), depth=int(d),
                                                kernels=kk, stride=int(s),
                                                norm=nm, dropout=dropout))
            ch_in = int(ch_out)
            dims.append(int(ch_out))

        self.level_names = ["L0"] + [f"L{i+1}" for i in range(n)]
        self.level_dims = dims
        acc, strides_out = 1, [1]
        for s in strides:
            acc *= int(s)
            strides_out.append(acc)
        self.level_strides = strides_out

    def forward(self, x):
        """Encode a window into one feature map per level."""
        feats = {"L0": self.stem(x)}
        prev = feats["L0"]
        for name, block in zip(self.level_names[1:], self.blocks):
            prev = block(prev)
            feats[name] = prev
        return feats


class TransformerEncoder1d(nn.Module):
    """Patch-transformer capacity-comparison arm (single level 'L0')."""

    def __init__(self, in_channels: int = 38, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, patch_len: int = 8,
                 dim_feedforward: int = 128, dropout: bool = True):
        super().__init__()
        self.patch_len = int(patch_len)
        self.patch_embed = nn.Conv1d(in_channels, d_model,
                                     kernel_size=self.patch_len,
                                     stride=self.patch_len)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward,
            dropout=0.1 if dropout else 0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.level_names = ["L0"]
        self.level_dims = [d_model]
        self.level_strides = [self.patch_len]

    @staticmethod
    def _sinusoidal_pe(tokens: int, d_model: int, device, dtype):
        """Standard sinusoidal position table (no learned parameters)."""
        position = torch.arange(tokens, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype)
                        * (-math.log(10000.0) / d_model))
        pe = torch.zeros(tokens, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe

    def forward(self, x):
        """Patch-embed, add positions, encode, return {'L0': (B, D, T)}."""
        import torch as _torch
        z = self.patch_embed(x)
        b, d, t = z.shape
        seq = z.transpose(1, 2) + self._sinusoidal_pe(t, d, z.device, z.dtype)
        out = self.norm(self.encoder(seq))
        return {"L0": out.transpose(1, 2).contiguous()}
