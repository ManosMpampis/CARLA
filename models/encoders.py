import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ResNetBlock

class ResNetEncoder(nn.Module):
    """Two-block time-domain encoder, single scale.

    ``enc_strides`` holds one stride per block (default [1, 1]: exact
    legacy behavior, length-preserving). Total stride S = prod(enc_strides)
    is exposed as ``level_strides=[S]`` so scorers map tokens back to input
    steps (token i covers input [i*S, (i+1)*S)). Phase-2 contract: training
    and inference windows must satisfy ``W % S == 0``; no crop/pad logic
    lives in the model by design (grilled Q8a).
    """

    def __init__(self, in_channels: int = 38, enc_channels=(32, 64),
                 kernels=(7, 5, 3), residual_kernel: int = 5, act: str = "relu",
                 norm: str = "batch", dropout: bool = True,
                 enc_strides=None):
        super().__init__()
        enc_channels = [in_channels] + list(enc_channels)
        n_blocks = len(enc_channels) - 1
        if enc_strides is None:
            enc_strides = [1] * n_blocks
        enc_strides = [int(s) for s in enc_strides]
        assert len(enc_strides) == n_blocks, \
            f"enc_strides {enc_strides} must match num blocks {n_blocks}"
        self.enc_strides = list(enc_strides)
        self.blocks = nn.Sequential(
            *[ResNetBlock(
                enc_channels[i], enc_channels[i + 1],
                kernels=kernels, residual_kernel=residual_kernel, act=act,
                norm=norm, dropout=dropout, stride=enc_strides[i])
              for i in range(n_blocks)]
        )
        self.output_dims = enc_channels[-1]
        # Channel ladder (incl. input) for the mirrored recon head.
        self.channel_ladder = [int(c) for c in enc_channels]
        self.total_stride = 1
        for s in self.enc_strides:
            self.total_stride *= int(s)

    def forward(self, x):
        """Encode (B, C, W) -> (B, D, W/S). Requires W % S == 0."""
        return self.blocks(x)