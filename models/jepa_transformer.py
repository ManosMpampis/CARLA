import math

import torch
import torch.nn as nn


class JEPATransformer(nn.Module):
    """Transformer token-encoder capacity-comparison arm.

    Patch-embeds the input window into tokens, adds sinusoidal positions,
    and runs a standard transformer encoder. Exposes the same level
    interface as PyramidEncoder (single coarse level 'L0') so the causal
    predictor wiring is unchanged. Kept per owner decision as a capacity
    comparison; never the default path.
    """

    def __init__(self, in_channels: int = 38, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, patch_len: int = 8,
                 dim_feedforward: int = 128, dropout: bool = True):
        super().__init__()
        self.patch_len = int(patch_len)
        self.patch_embed = nn.Conv1d(in_channels, d_model,
                                     kernel_size=self.patch_len,
                                     stride=self.patch_len)
        self.pos_encoding = self._make_sinusoidal_pe
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward,
            dropout=0.1 if dropout else 0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.level_names = ["L0"]
        self.level_dims = [d_model]
        self.level_strides = [self.patch_len]

    @staticmethod
    def _make_sinusoidal_pe(tokens: int, d_model: int, device, dtype):
        position = torch.arange(tokens, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=dtype)
                        * (-math.log(10000.0) / d_model))
        pe = torch.zeros(tokens, d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe

    def forward(self, x: torch.Tensor) -> dict:
        z = self.patch_embed(x)                       # (B, D, T)
        b, d, t = z.shape
        seq = z.transpose(1, 2) + self._make_sinusoidal_pe(
            t, d, z.device, z.dtype)
        out = self.norm(self.encoder(seq))            # (B, T, D)
        return {"L0": out.transpose(1, 2).contiguous()}
