import torch.nn as nn

from models.blocks import ResnetConvBlock


class ResNetEncoder(nn.Module):
    """Two-block time-domain encoder, single scale.
    """

    def __init__(self, in_ch: int = 38, out_ch=(32, 64),
                 kernels=(7, 5, 3), residual_kernel: int = 5,
                 norm: str = "batch", act: str = "gelu", dropout: bool = True,
                 strides=None):
        super().__init__()
        
        out_ch = [in_ch] + list(out_ch)
        n_blocks = len(out_ch) - 1

        if strides is None:
            strides = [1] * n_blocks
        strides = [int(s) for s in strides]

        assert len(strides) == n_blocks, \
            f"enc_strides {strides} must match num blocks {n_blocks}"
        

        self.blocks = nn.Sequential(
            *[ResnetConvBlock(
                out_ch[i], out_ch[i + 1],
                kernels=kernels, residual_kernel=residual_kernel,
                norm=norm, act=act, dropout=dropout, stride=int(strides[i]))
              for i in range(n_blocks)]
        )
        
        self.output_dims = out_ch[-1]
        # Channel ladder (incl. input) for the mirrored recon head.
        self.channel_ladder = out_ch
        self.strides = list(strides)

        total = 1
        for s in self.strides:
            total *= int(s)

        self.kernels = list(kernels)
        self.n_blocks = int(n_blocks)
        self.total_stride = int(total)
        self.level_dims = [int(self.output_dims)]
        self.level_strides = [int(total)]

    def forward(self, x):
        """Encode (B, C, W) -> (B, D, W/S). Requires W % S == 0."""
        return self.blocks(x)