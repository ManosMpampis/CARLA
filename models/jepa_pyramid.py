import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convolutions import _init_weights


def cumulative_strides(chain) -> list:
    """Running product of a stride chain starting with the stem's own 1."""
    out = []
    acc = 1
    for s in chain:
        acc *= s
        out.append(acc)
    return out


class StridedConvBlock(nn.Module):
    """Conv-norm-GELU block with true temporal downsampling.

    Not built on ConvBlock/Conv1dSamePadding: that helper derives padding
    assuming stride 1, which silently cancels stride>1 downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, dropout: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dout = nn.Dropout(0.2) if dropout else nn.Identity()
        _init_weights(self.conv)
        _init_weights(self.norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dout(self.act(self.norm(self.conv(x))))


class PyramidLevel(nn.Module):
    """Downsamples the time axis and refines features at the new scale."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, kernel_size: int = 5):
        super().__init__()
        self.downsample = StridedConvBlock(in_channels, out_channels, kernel_size, stride=stride)
        self.refine = StridedConvBlock(out_channels, out_channels, kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.downsample(x))


class PyramidEncoder(nn.Module):
    """Multi-scale convolutional encoder.

    Input (B, C, W) -> one latent map per level: {name: (B, D_l, T_l)}.
    Each level covers a different sub-window length of the input window,
    giving the multi-scale axis used for sub-window anomaly scoring.
    """

    def __init__(
        self,
        in_channels: int = 38,
        stem_channels: int = 32,
        level_channels: tuple = (32, 64, 96),
        kernel_size: int = 5,
        strides: tuple = (2, 2, 2),
        dropout: bool = True,
    ):
        super().__init__()
        assert len(level_channels) == len(strides)
        self.stem = StridedConvBlock(in_channels, stem_channels, kernel_size=7, stride=1, dropout=dropout)
        levels = []
        ch_in = stem_channels
        for i, ch_out in enumerate(level_channels):
            levels.append((
                f"L{i+1}",
                PyramidLevel(ch_in, ch_out, stride=strides[i], kernel_size=kernel_size),
                ch_out,
            ))
            ch_in = ch_out
        self.level_names = ["L0"] + [name for name, _, _ in levels]
        self.level_dims = [stem_channels] + [dim for _, _, dim in levels]
        self.level_strides = cumulative_strides([1] + list(strides))
        self.levels = nn.ModuleList(mod for _, mod, _ in levels)

    def forward(self, x: torch.Tensor) -> dict:
        feats = {"L0": self.stem(x)}
        prev = "L0"
        for name, level in zip(self.level_names[1:], self.levels):
            feats[name] = level(feats[prev])
            prev = name
        return feats


class CausalConv1d(nn.Conv1d):
    """Left-padded causal convolution: output[t] sees inputs <= t only."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__(
            in_channels, out_channels, kernel_size,
            padding=0, dilation=dilation,
        )
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(F.pad(x, (pad, 0)))


class CausalTCNPredictor(nn.Module):
    """Predicts future latents at one pyramid level from strictly past latents.

    Output (B, horizons, D, T): entry [:, k-1, :, t] predicts the latent at
    position t+k of the same level, using only positions <= t.
    Fully parallel over t during training; single pass at inference.
    """

    def __init__(self, dim: int, horizons: int = 2, hidden: int | None = None,
                 dilations: tuple = (1, 2, 4), kernel_size: int = 3):
        super().__init__()
        hidden = hidden or dim
        blocks = []
        ch_in = dim
        for d in dilations:
            blocks.extend([
                CausalConv1d(ch_in, hidden, kernel_size, dilation=d),
                nn.GELU(),
            ])
            ch_in = hidden
        self.body = nn.Sequential(*blocks)
        self.head = CausalConv1d(hidden, horizons * dim, kernel_size=1)
        self.horizons = horizons
        self.dim = dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.body(z)
        out = self.head(h)
        b = out.size(0)
        return out.view(b, self.horizons, self.dim, -1)


class GRUPredictor(nn.Module):
    """Recurrent causal predictor alternative to the TCN.

    The GRU consumes latents left-to-right, so position t only sees <= t;
    a linear head emits all horizon predictions per position.
    """

    def __init__(self, dim: int, horizons: int = 2, hidden: int | None = None, num_layers: int = 1):
        super().__init__()
        hidden = hidden or dim
        self.gru = nn.GRU(dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, horizons * dim)
        self.horizons = horizons
        self.dim = dim
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_weights(module)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        seq = z.transpose(1, 2)  # (B, T, D)
        h, _ = self.gru(seq)
        out = self.head(h)  # (B, T, horizons * D)
        out = out.view(b, -1, self.horizons, self.dim).permute(0, 2, 3, 1)
        return out.contiguous()  # (B, horizons, D, T)
