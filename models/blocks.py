"""Shared convolutional building blocks (configurable, documented).

All blocks are plain classes with methods. Kernel/stride/depth choices
come from config; scalar args broadcast over the block's convolutions.
"""
import torch.nn as nn
import torch.nn.functional as F

from models.convolutions import _init_weights


def make_norm(kind: str, channels: int) -> nn.Module:
    """Build a 1D norm layer by config name."""
    kind = str(kind).lower()
    if kind in ("none", "no", "identity"):
        return nn.Identity()
    if kind == "layer":
        # LayerNorm over (C, T) needs no time dim; applied per-token below.
        return nn.GroupNorm(1, channels)
    if kind == "instance":
        return nn.InstanceNorm1d(channels, affine=True)
    return nn.BatchNorm1d(channels)


class ConvBlock1d(nn.Module):
    """One Conv1d + norm + GELU + dropout with true striding.

    Unlike the legacy same-padding helper (stride-1 assumption), padding
    here preserves length only when stride == 1.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5,
                 stride: int = 1, norm: str = "batch", dropout: bool = True):
        super().__init__()
        # Padding preserves length at stride 1; stride > 1 downsamples by design.
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              padding=kernel // 2)
        self.norm = make_norm(norm, out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.2) if dropout else nn.Identity()
        _init_weights(self.conv)
        _init_weights(self.norm)

    def forward(self, x):
        """Apply convolution, normalization, activation, dropout."""
        return self.drop(self.act(self.norm(self.conv(x))))


class StackedConvBlock(nn.Module):
    """N convolutions at fixed resolution: downsample once, refine rest.

    Args:
        in_ch: input channels. out_ch: output channels.
        depth: total convs in this block (>= 1).
        kernels: kernel per conv (scalar broadcasts).
        stride: applied to the FIRST conv only; rest are stride-1.
    """

    def __init__(self, in_ch: int, out_ch: int, depth: int = 2,
                 kernels=5, stride: int = 2,
                 norm: str = "batch", dropout: bool = True):
        super().__init__()
        ks = list(kernels) if isinstance(kernels, (list, tuple)) else [kernels] * depth
        assert len(ks) == depth, "kernels must match depth"
        layers = []
        ch = in_ch
        for i in range(depth):
            layers.append(ConvBlock1d(ch, out_ch, kernel=int(ks[i]),
                                      stride=stride if i == 0 else 1,
                                      norm=norm, dropout=dropout))
            ch = out_ch
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Run the stacked convolutions in order."""
        return self.layers(x)


class CausalConv1d(nn.Conv1d):
    """Left-padded causal convolution: output[t] sees inputs <= t only."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__(in_ch, out_ch, kernel, padding=0, dilation=dilation)
        _init_weights(self)

    def forward(self, x):
        """Pad on the left so future positions never leak in."""
        pad = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(F.pad(x, (pad, 0)))
