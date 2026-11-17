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

def make_act(kind: str) -> nn.Module:
    """Build an activation layer by config name."""
    kind = str(kind).lower()
    if kind in ("none", "no", "identity"):
        return nn.Identity()
    if kind == "relu":
        return nn.ReLU()
    if kind == "gelu":
        return nn.GELU()
    else:
        raise ValueError("No activation layer with name {kind} has been configured")

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


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = "zeros", device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        _init_weights(self)

    def forward(self, input):

        kernel, dilation, stride = self.weight.size()[2], self.dilation[0], self.stride[0]
        l_out = l_in = input.size()[2]
        padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
        input = F.pad(input, [0, padding % 2])
        padding = (padding // 2,)
        output = F.conv1d(input=input, weight=self.weight, bias=self.bias, stride=stride,
                        padding=padding, dilation=dilation, groups=self.groups)
        return output

class Conv2dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = "zeros", device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        _init_weights(self)
        
    def forward(self, input):

        kernel, dilation, stride = self.weight.size()[2], self.dilation[0], self.stride[0]
        l_out = l_in = input.size()[2]
        padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
        input = F.pad(input, [0, padding % 2, 0, padding % 2])
        padding = (padding // 2, 0, padding // 2, 0)
        output = F.conv1d(input=input, weight=self.weight, bias=self.bias, stride=stride,
                        padding=padding, dilation=dilation, groups=self.groups)
        return output

class CausalConv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):

        kernel, dilation, stride = self.weight.size()[2], self.dilation[0], self.stride[0]
        l_out = l_in = input.size()[2]
        padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
        input = F.pad(input, (padding, 0))
        output = F.conv1d(input=input, weight=self.weight, bias=self.bias, stride=stride,
                        padding=0, dilation=dilation, groups=self.groups)
        return output


class ConvBlock1d(nn.Module):
    """One Conv1d + norm + act + dropout with true striding.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5,
                 stride: int = 1, dilation: int = 1,
                 act: str = "gelu", norm: str = "batch", dropout: float = 0.0, causal: bool = False):
        super().__init__()
        if causal:
            self.conv = CausalConv1dSamePadding(in_channels=in_ch,
                                                  out_channels=out_ch,
                                                  kernel_size=kernel,
                                                  stride=stride,
                                                  dilation=dilation)
        else:
            self.conv = Conv1dSamePadding(in_channels=in_ch,
                                      out_channels=out_ch,
                                      kernel_size=kernel,
                                      stride=stride,
                                      dilation=dilation)
        self.norm = make_norm(norm, out_ch)
        self.act = make_act(act)
        self.drop = nn.Dropout(dropout) if dropout>0.0 else nn.Identity()
        _init_weights(self.conv)
        _init_weights(self.norm)

    def forward(self, x):
        """Apply convolution, normalization, activation, dropout."""
        return self.drop(self.act(self.norm(self.conv(x))))


class ConvBlock2d(nn.Module):
    """One Conv2d + norm + act + dropout with true striding.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5,
                 stride: int = 1, dilation: int = 1,
                 act: str = "gelu", norm: str = "batch", dropout: float = 0.0, causal: bool = False):
        super().__init__()
        self.conv = Conv2dSamePadding(in_channels=in_ch,
                                      out_channels=out_ch,
                                      kernel_size=kernel,
                                      stride=stride,
                                      dilation=dilation)
        self.norm = make_norm(norm, out_ch)
        self.act = make_act(act)
        self.drop = nn.Dropout(dropout) if dropout>0.0 else nn.Identity()
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

    def __init__(self, in_ch: int, out_ch: list,
                 kernels: list =[5, 5], stride: list = [2], dilation: list = [1],
                 act: str = "gelu", norm: str = "batch", dropout: float = 0.1, causal: bool = False):
        super().__init__()
        ks = list(kernels)
        st = list(stride)
        dl = list(dilation)
        ch = [in_ch] + out_ch
        if len(st) == 1: st = st * len(ks)
        if len(dl) == 1: dl = dl * len(ks)
        assert len(ks) == len(st) == len(dl) == len(ch)
        self.layers = nn.Sequential(
            *[
                ConvBlock1d(ch[i], ch[i+1], kernel=ks[i], stride=st[i], dilation=dl[i], 
                            act=act, norm=norm, dropout=dropout, causal=causal) for i in len(ks)
            ]
        )

    def forward(self, x):
        """Run the stacked convolutions in order."""
        return self.layers(x)

    
class ResnetConvBlock(nn.Module):
    """Residual convolution block.

    Args:
        in_ch: input channels. out_ch: output channels.
        depth: total convs in this block (>= 1).
        kernels: kernel per conv (scalar broadcasts).
        stride: applied to the FIRST conv only; rest are stride-1.
    """

    def __init__(self, in_ch: int, out_ch: list,
                 kernels: list =[7, 5, 3], stride: int = 1, residual_kernel: int = 5,
                 act: str = "gelu", norm: str = "batch", dropout: float = 0.1, causal: bool = False):
        super().__init__()
        stride = [stride] + ([1] * (len(kernels)-1))
        assert len(kernels) > 1, "residual block should have more than one kernel in their main arm"
        self.main = nn.Sequential(
            StackedConvBlock(in_ch, out_ch[:-1], kernels[:-1], stride[:-1], 1, act, norm, dropout, causal),
            ConvBlock1d(out_ch[-2], out_ch[-1], residual_kernel[-1], stride[-1], 1, "none", norm, dropout, causal)
            )
        self.residual = ConvBlock1d(in_ch, out_ch, residual_kernel, stride[0], 1, "none", norm, dropout, causal)
        self.act = make_act(act)

    def forward(self, x):
        """Run the stacked convolutions in order."""
        return self.act(self.main(x) + self.residual(x))


class CrossAttention(nn.Module):
    """Scaled cross attention and reconstruction, without a Query residual bypass."""

    def __init__(self, num_heads=1,
                 dropout=0.1, position_encoding="none",
                 position_on="all", position_scale=1.0):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("attention dropout must be in [0, 1)")
        if position_encoding not in ("none", "sinusoidal"):
            raise ValueError("position_encoding must be none or sinusoidal")
        if position_on not in ("query", "key", "value", "qk", "qv", "kv", "all"):
            raise ValueError("position_on must be query, key, value, qk, qv, kv, or all")
        if (isinstance(position_scale, bool) or not isinstance(position_scale, (int, float))
                or not math.isfinite(position_scale) or position_scale < 0):
            raise ValueError("position_scale must be finite and nonnegative")
        self.num_heads = num_heads
        self.dropout = dropout
        self.position_encoding = position_encoding
        self.position_on = position_on
        self.position_scale = float(position_scale)

        self.q_enc = nn.Identity()
        self.k_enc = nn.Identity()
        self.v_enc = nn.Identity()
        if position_encoding == "sinusoidal":
            if position_on in ("query", "qk", "qv", "all"):
                self.q_enc = self._with_position
            if position_on in ("key", "qk", "kv", "all"):
                self.k_enc = self._with_position
            if position_on in ("value", "qv", "kv", "all"):
                self.v_enc = self._with_position
        
    def _with_position(self, z):
        """Add fixed sinusoids in feature-token units, supporting any length/width."""
        channels, length = z.shape[1:]
        positions = torch.arange(0, length, device=z.device, dtype=z.dtype)
        frequencies = torch.exp(torch.arange(0, channels, 2, device=z.device,
                                            dtype=z.dtype) * (-math.log(10000.0) / channels))
        angles = frequencies[:, None] * positions[None, :]
        encoding = torch.empty(channels, length, device=z.device, dtype=z.dtype)
        encoding[0::2] = angles.sin()
        encoding[1::2] = angles[:channels // 2].cos()
        return z + self.position_scale * encoding.to(z.dtype).unsqueeze(0)

    def forward(self, kv, q):
        # Enrich attention addressing only. Keep content Q/K/V unchanged for
        # SIGReg and the clean Query reconstruction target.
        query = self.q_enc(kv["query"])
        key = self.k_enc(q)
        value = self.v_enc(kv["value"])

        def heads(z):
            b, c, t = z.shape
            return z.reshape(b, self.num_heads, c // self.num_heads, t).transpose(-1, -2)

        attended = F.scaled_dot_product_attention( heads(query), heads(key), heads(value),
            dropout_p=self.dropout if self.training else 0.0)
        
        b, _, t, _ = attended.shape
        attended = attended.transpose(-1, -2).reshape(b, -1, t)
        return attended
