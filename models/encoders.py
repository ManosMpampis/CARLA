"""Time-domain feature extractors with explicit channel and stride metadata."""

import math

from torch import nn

from models.blocks import CausalConv1dBlock, Conv1dBlock, ResNetBlock, _init_weights, _per_layer


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 38, enc_channels=(32, 64),
                 kernels=(7, 5, 3), residual_kernel: int = 5, act: str = "gelu",
                 norm: str = "batch", dropout: float = 0.1, enc_strides=None,
                 output_act=None, latent_output="legacy"):
        super().__init__()
        channels = [in_channels, *enc_channels]
        self.channel_ladder = channels
        self.input_dims = in_channels
        if latent_output not in ("legacy", "unrestricted"):
            raise ValueError("latent_output must be legacy or unrestricted")
        self.latent_output = latent_output
        self.output_dims = channels[-1]
        count = len(enc_channels)
        if not count:
            raise ValueError("enc_channels must contain at least one width")
        self.enc_strides = _per_layer(1 if enc_strides is None else enc_strides,
                                      count, "enc_strides")
        self.blocks = nn.Sequential(*[
            ResNetBlock(channels[index], channels[index + 1], kernels=kernels,
                        residual_kernel=residual_kernel,
                        stride=stride, norm=norm, act=act, dropout=dropout,
                        output_act=output_act if index == count - 1 else None,
                        unrestricted_output=latent_output == "unrestricted" and index == count - 1)
            for index, stride in enumerate(self.enc_strides)
        ])
        self.total_stride = math.prod(self.enc_strides)
        self.level_dims = [self.output_dims]
        self.level_strides = [self.total_stride]

    def forward(self, x):
        return self.blocks(x)


class Encoder(nn.Module):
    """Convolution stack with a linear output projection."""

    block = Conv1dBlock

    def __init__(self, in_dim, hidden_dims, out_dims, kernels=3, stride=1,
                 dilation=1, norm="batch", act="gelu", dropout=0.0):
        super().__init__()
        hidden_dims = list(hidden_dims)
        count = len(hidden_dims)
        kernels = _per_layer(kernels, count, "kernels")
        strides = _per_layer(stride, count, "stride")
        dilations = _per_layer(dilation, count, "dilation")
        widths = [in_dim, *hidden_dims]
        self.input_dims = in_dim
        self.output_dims = out_dims
        self.total_stride = math.prod(strides)
        self.channel_ladder = [*widths, out_dims]
        self.enc_strides = [*strides, 1]
        self.net = nn.Sequential(*[
            self.block(a, b, kernel=kernel, stride=step, dilation=rate,
                       norm=norm, act=act, dropout=dropout)
            for a, b, kernel, step, rate in zip(widths, widths[1:], kernels, strides, dilations)
        ], nn.Conv1d(widths[-1], out_dims, 1))
        _init_weights(self.net[-1])
        self.receptive_field = 1 + sum(
            (kernel - 1) * rate * math.prod(strides[:index])
            for index, (kernel, rate) in enumerate(zip(kernels, dilations))
        )

    def forward(self, x):
        return self.net(x)


class TCNEncoder(Encoder):
    """Left-padded temporal encoder. Default normalization is time-local."""

    block = CausalConv1dBlock

    def __init__(self, *args, norm="none", **kwargs):
        super().__init__(*args, norm=norm, **kwargs)


class IdentityEncoder(nn.Identity):
    """Identity feature path retaining the dimensional interface of encoders."""

    def __init__(self, in_dim):
        super().__init__()
        self.input_dims = self.output_dims = int(in_dim)
        self.total_stride = 1
        self.channel_ladder = [int(in_dim)]
        self.enc_strides = []
