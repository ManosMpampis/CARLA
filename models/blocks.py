"""Shared convolutional building blocks (configurable, documented).

All blocks are plain classes with methods. Kernel/stride/depth choices
come from config; scalar args broadcast over the block's convolutions.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module):
    """Initialize weights and biases for all supported layer types."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        # Kaiming (He) initialization for convolutional layers
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        # Kaiming initialization for linear layers
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d, nn.LayerNorm)):
        # Batch norm initialization
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm1d):
        # Instance norm initialization
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        # Layer norm initialization
        module.weight=torch.ones(module.num_features)
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def make_norm1d(kind: str, channels: int) -> nn.Module:
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


def make_act(kind: str) -> nn.Module:
    """Build a activation layer by config name."""
    kind = str(kind).lower()
    if kind in ("none", "no", "identity"):
        return nn.Identity()
    if kind == "relu":
        return nn.ReLU()
    if kind == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation kind: {kind}")


class Conv1dSamePadding(nn.Conv1d):
    """Centered padding with output length ceil(input length / stride)."""

    def forward(self, x):
        extent = self.dilation[0] * (self.kernel_size[0] - 1)
        left = extent // 2
        return F.conv1d(F.pad(x, (left, extent - left)), self.weight, self.bias,
                        stride=self.stride, dilation=self.dilation, groups=self.groups)


class Conv1dBlock(nn.Module):
    """One Conv1d + norm + activation + dropout with true striding."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, stride: int = 1,
                 dilation: int = 1, norm: str = "batch", act: str = "gelu",
                 dropout: float = 0.0):
        super().__init__()
        extent = dilation * (kernel - 1)
        # Keep the historical Conv1d path and state keys for odd kernels.
        conv = nn.Conv1d if extent % 2 == 0 else Conv1dSamePadding
        self.conv = conv(in_ch, out_ch, kernel, stride=stride, dilation=dilation,
                         padding=extent // 2 if conv is nn.Conv1d else 0)
        self.norm = make_norm1d(norm, out_ch)
        self.act = make_act(act)
        self.drop = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        _init_weights(self.conv)
        _init_weights(self.norm)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))


class Conv2dSamePadding(nn.Conv2d):
    """Centered 2D padding with true striding on both axes."""

    def forward(self, x):
        height, width = (rate * (size - 1) for rate, size in zip(self.dilation, self.kernel_size))
        left, top = width // 2, height // 2
        x = F.pad(x, (left, width - left, top, height - top))
        return F.conv2d(x, self.weight, self.bias, stride=self.stride,
                        dilation=self.dilation, groups=self.groups)


class Conv2dBlock(nn.Module):
    """Conv2d + BatchNorm2d + act + dropout block for the TF-grid neck."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1,
                 dilation: int = 1, act: str = "gelu", dropout: float = 0.0):
        super().__init__()
        extent = dilation * (kernel - 1)
        conv = nn.Conv2d if extent % 2 == 0 else Conv2dSamePadding
        self.net = nn.Sequential(
            conv(in_ch, out_ch, kernel, stride=stride, dilation=dilation,
                 padding=extent // 2 if conv is nn.Conv2d else 0),
            nn.BatchNorm2d(out_ch),
            make_act(act),
            nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _init_weights(m)
            elif isinstance(m, nn.BatchNorm2d):
                _init_weights(m)

    def forward(self, x):
        """Apply conv block."""
        return self.net(x)


class StackedConv1dBlock(nn.Module):
    """N convolutions at fixed resolution: downsample once, refine rest.

    Args:
        in_ch: input channels. out_ch: output channels.
        kernels: per-layer kernel sizes, or one kernel for a single convolution.
        stride: one stride for the first convolution, or a stride per layer.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernels=(5, 5), stride: int = 2, dilation: int = 1,
                 norm: str = "batch", act: str = "gelu", dropout: float = 0.1):
        super().__init__()
        ks = [kernels] if isinstance(kernels, int) else list(kernels)
        if not ks:
            raise ValueError("kernels must contain at least one layer")
        st = [stride] + [1] * (len(ks) - 1) if isinstance(stride, int) else stride
        st = _per_layer(st, len(ks), "stride")
        dl = _per_layer(dilation, len(ks), "dilation")
        layers = []
        ch = in_ch
        for i in range(len(ks)):
            layers.append(Conv1dBlock(ch, out_ch, kernel=ks[i],
                                      stride=st[i], dilation=dl[i],
                                      norm=norm, act=act, dropout=dropout))
            ch = out_ch
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Run the stacked convolutions in order."""
        return self.layers(x)


class ResNetBlock(nn.Module):
    """One ResNet block: 3 main convs [7,5,3] + residual conv k=5.

    Stride (default 1, length-preserving) is applied to the FIRST main
    conv and to the residual branch so both paths downsample identically.
    Stride > 1 downsamples by design (see ConvBlock1d); the caller must
    keep ``wsz % prod(strides) == 0`` so the mirrored recon head inverts
    shapes exactly with no crop logic (Phase-2 contract).
    """

    def __init__(self, in_ch: int, out_ch: int, kernels=(7, 5, 3),
                 residual_kernel: int = 5, norm: str = "batch", act: str = "gelu",
                 dropout: float = 0.0, stride: int = 1, output_act=None,
                 unrestricted_output: bool = False):
        super().__init__()
        self.stride = int(stride)
        ks = list(kernels)
        chs = [in_ch] + [out_ch] * len(ks)
        self.main = nn.Sequential(
            *[Conv1dBlock(chs[i], chs[i + 1], kernel=k,
                          stride=self.stride if i == 0 else 1,
                          norm=norm, dropout=dropout,
                          act="identity" if unrestricted_output and i == len(ks) - 1 else act)
              for i, k in enumerate(ks)]
        )
        self.residual = Conv1dBlock(in_ch, out_ch, kernel=int(residual_kernel),
                                    stride=self.stride, norm=norm,
                                    act="identity" if unrestricted_output else act,
                                    dropout=dropout)
        self.act = make_act("identity" if unrestricted_output else act if output_act is None else output_act)

    def forward(self, x):
        """Add residual branch to main branch, then activate."""
        return self.act(self.main(x) + self.residual(x))


class CausalConv1d(nn.Conv1d):
    """Left-padded causal convolution: output[t] sees inputs <= t only."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, dilation: int = 1):
        super().__init__(in_ch, out_ch, kernel, padding=0, dilation=dilation, stride=stride)
        self.pad = self.dilation[0] * (self.kernel_size[0] - 1)
        _init_weights(self)

    def forward(self, x):
        """Pad on the left so future positions never leak in."""
        return super().forward(F.pad(x, (self.pad, 0)))


class CausalConv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5, dilation: int = 1, stride: int = 1, 
                 norm: str = "batch", act: str = "gelu", dropout: float = 0.0):
        super().__init__() 
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel, dilation=dilation, stride=stride),
            make_norm1d(norm, out_ch),
            make_act(act),
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                _init_weights(m)
            elif isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm1d, nn.LayerNorm)):
                _init_weights(m)
    
    def forward(self, x):
        """Apply convolution, normalization, activation, dropout."""
        return self.net(x)


def _per_layer(values, count, name):
    values = [values] * count if isinstance(values, int) else list(values)
    if len(values) != count or any(type(value) is not int or value < 1 for value in values):
        raise ValueError(f"{name} needs {count} positive integer values")
    return values


class CrossAttention(nn.Module):
    """Scaled cross attention and reconstruction."""

    def __init__(self, head=None, qk_channels=8, value_channels=8, num_heads=1,
                 dropout=0.0, input_projection=None, position_encoding="none",
                 position_on="both", position_scale=1.0, diagonal_mask=False):
        super().__init__()
        if (min(num_heads, qk_channels, value_channels) < 1
                or qk_channels % num_heads or value_channels % num_heads):
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
        if not isinstance(diagonal_mask, bool):
            raise ValueError("diagonal_mask must be true or false")
        self.diagonal_mask = diagonal_mask
        self.head = head if head is not None else nn.Identity()
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

    def forward(self, q, k, v, *, query_start=0, key_start=0):
        if (q.ndim != 3 or k.ndim != 3 or v.ndim != 3
                or q.size(0) != k.size(0) or k.size(0) != v.size(0)
                or q.size(1) != k.size(1) or k.size(-1) != v.size(-1)
                or q.size(1) % self.num_heads or v.size(1) % self.num_heads):
            raise ValueError("attention requires matching Q/K widths and K/V lengths")
        if self.position_encoding != "none" and self.position_scale:
            if self.position_on in ("query", "both"):
                q = self._with_position(q, query_start)
            if self.position_on in ("key", "both"):
                k = self._with_position(k, key_start)

        def heads(z):
            b, c, t = z.shape
            return z.reshape(b, self.num_heads, c // self.num_heads, t).transpose(-1, -2)

        attention_mask = None
        if self.diagonal_mask:
            # SDPA boolean masks use True for ALLOWED pairs. These are local
            # crop indices: allow only keys strictly earlier than each query.
            # The first query has no allowed keys; SDPA returns zero there.
            query_indices = torch.arange(q.size(-1), device=q.device)
            key_indices = torch.arange(k.size(-1), device=q.device)
            attention_mask = key_indices[None, :] < query_indices[:, None]

        attended = F.scaled_dot_product_attention(
            heads(q), heads(k), heads(v),
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0)
        b, _, t, _ = attended.shape
        attended = attended.transpose(-1, -2).reshape(b, -1, t)
        return self.head(self.input_projection(attended))


class Identity(nn.Module):
    """Identity layer for no-op backbones, actions, decoders, etc."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
