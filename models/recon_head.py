"""Mirrored reconstruction head for the steered ResNet encoder (Phase 2).

Grilled contract (Q8/Q8a/Q8b):
- N ConvTranspose1d layers mirroring the N ResNet blocks in reverse, with
  the same stride/dilation rate per block, then a single 1x1 conv to the
  input channel count C. Default 2-block encoder -> 3 convs total.
- Fully convolutional (no Linear, no fixed pooling): any input length W
  with ``W % S == 0`` reconstructs exactly to (B, C, W) with no crop/pad
  logic in the model. Odd lengths are a caller error (asserted).
- Channels mirror the encoder ladder: encoder [C0, D1, ..., DN] ->
  transposes [DN -> D_{N-1} -> ... -> D1 -> D1] then 1x1 D1 -> C0.
"""

import torch.nn as nn

from models.blocks import make_norm
from models.convolutions import _init_weights


class ReconUpsampleBlock(nn.Module):
    """One ConvTranspose1d + norm + GELU + dropout (mirror of a strided block)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 kernel: int = 5, dilation: int = 1,
                 norm: str = "batch", dropout: float = 0.0):
        super().__init__()
        stride = int(stride)
        # Invert stride-s downsampling exactly when lengths divide evenly:
        #   s=1 -> k=5,pad=2,out_pad=0 preserves length;
        #   s>1 -> k=4,pad=1,out_pad=s-2 multiplies length by s exactly
        # (dilation 1; strided+dilated combos are out of contract).
        if stride == 1:
            kernel, pad, out_pad = 5, 2, 0
        else:
            assert int(dilation) == 1, "strided+dilated head out of contract"
            kernel, pad, out_pad = 4, 1, int(stride) - 2
        self.deconv = nn.ConvTranspose1d(
            int(in_ch), int(out_ch), kernel_size=int(kernel),
            stride=stride, padding=int(pad), output_padding=int(out_pad),
            dilation=int(dilation))
        self.norm = make_norm(norm, int(out_ch))
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        _init_weights(self.deconv)
        _init_weights(self.norm)

    def forward(self, x):
        """Upsample one mirror step."""
        return self.drop(self.act(self.norm(self.deconv(x))))


class MirroredReconHead(nn.Module):
    """N transposes (reverse strides) + 1x1 conv to input channels."""

    def __init__(self, channel_ladder, enc_strides, norm: str = "batch",
                 dropout: float = 0.0, dilations=None):
        super().__init__()
        ladder = [int(c) for c in channel_ladder]
        assert len(ladder) >= 2, "ladder needs [C0, D1, ...]"
        strides = [int(s) for s in enc_strides]
        assert len(strides) == len(ladder) - 1
        if dilations is None:
            dilations = [1] * len(strides)
        dilations = [int(d) for d in dilations]
        assert len(dilations) == len(strides)

        # Hidden dims [D1..DN]; transposes walk DN -> .. -> D1 -> D1.
        hiddens = ladder[1:]
        rev_strides = list(reversed(strides))
        rev_dilations = list(reversed(dilations))
        in_dims = list(reversed(hiddens))
        out_dims = list(reversed(hiddens[:-1])) + [hiddens[0]]
        assert len(in_dims) == len(strides) and len(out_dims) == len(strides)

        self.blocks = nn.Sequential(*[
            ReconUpsampleBlock(i, o, stride=s, dilation=d,
                               norm=norm, dropout=dropout)
            for i, o, s, d in zip(in_dims, out_dims, rev_strides, rev_dilations)
        ])
        # Final 1x1 projection to input channels (no activation: the target
        # is standardized telemetry covering the full real range).
        self.to_input = nn.Conv1d(hiddens[0], ladder[0], kernel_size=1)
        _init_weights(self.to_input)
        self.out_channels = int(ladder[0])
        self.total_stride = 1
        for s in strides:
            self.total_stride *= int(s)

    def forward(self, z):
        """Map latents (B, DN, W/S) -> recon (B, C0, W)."""
        h = self.blocks(z)
        return self.to_input(h)


def build_mirrored_head(encoder, norm: str = "batch", dropout: float = 0.0,
                        dilations=None):
    """Build a MirroredReconHead from a SteeredResNetEncoder instance."""
    ladder = list(getattr(encoder, "channel_ladder",
                          [encoder.blocks[0].main[0].conv.in_channels,
                           encoder.output_dims]))
    strides = list(getattr(encoder, "enc_strides", [1] * (len(ladder) - 1)))
    enc_norm = getattr(encoder.blocks[0].main[0], "norm", None)
    _ = enc_norm
    return MirroredReconHead(ladder, strides, norm=norm, dropout=dropout,
                             dilations=dilations)
