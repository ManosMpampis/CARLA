"""Masked-part predictors and causal forecast-shaped predictors.

Default is masked reconstruction (I-JEPA style): a predictor class that
reads visible tokens on BOTH sides of a hole and reconstructs the masked
tokens' latents. The causal TCN/GRU classes (a part predicts offset-k
latents from past only) stay as tournament arms.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import CausalConv1d
from models.convolutions import _init_weights


class TCNPredictor(nn.Module):
    """Dilated causal-TCN predictor, one instance per pyramid level."""

    def __init__(self, dim: int, horizons: int = 2, hidden=None,
                 dilations=(1, 2, 4), kernel_size: int = 3):
        super().__init__()
        hidden = hidden or dim
        layers, ch = [], dim
        for d in dilations:
            layers += [CausalConv1d(ch, hidden, kernel_size, dilation=d), nn.GELU()]
            ch = hidden
        self.body = nn.Sequential(*layers)
        self.head = CausalConv1d(hidden, horizons * dim, 1)
        self.horizons, self.dim = horizons, dim

    def forward(self, z, mask_pos=None):
        """Map (B, D, T) to (B, H, D, T) part predictions."""
        out = self.head(self.body(z))
        return out.view(out.size(0), self.horizons, self.dim, -1)


class GRUPredictor(nn.Module):
    """Recurrent causal alternative to the TCN predictor."""

    def __init__(self, dim: int, horizons: int = 2, hidden=None, num_layers: int = 1):
        super().__init__()
        hidden = hidden or dim
        self.gru = nn.GRU(dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, horizons * dim)
        self.horizons, self.dim = horizons, dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_weights(m)

    def forward(self, z, mask_pos=None):
        """Left-to-right GRU; position t sees inputs <= t only."""
        h, _ = self.gru(z.transpose(1, 2))
        out = self.head(h).view(z.size(0), -1, self.horizons, self.dim)
        return out.permute(0, 2, 3, 1).contiguous()


class CondPredictor(nn.Module):
    """Action-conditioned wrapper (closed-loop arm only).

    Injects a per-window action vector via zero-initialized scale/shift
    (AdaLN trick): at init the wrapper behaves exactly like the base
    predictor, then learns conditioning gradually.
    """

    def __init__(self, base: nn.Module, dim: int, action_dim: int = 16):
        super().__init__()
        self.base = base
        self.to_scale = nn.Linear(action_dim, dim)
        self.to_shift = nn.Linear(action_dim, dim)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, z, action=None, mask_pos=None):
        """Predict with optional action; None disables conditioning."""
        if action is None:
            return self.base(z, mask_pos)
        # Broadcast per-window scale/shift over time.
        scale = 1 + self.to_scale(action).unsqueeze(-1)
        shift = self.to_shift(action).unsqueeze(-1)
        return self.base(z * scale + shift, mask_pos)


def build_predictor(kind: str, dim: int, horizons: int, hidden=None, **kwargs):
    """Factory honoring the predictor registry names (masked|tcn|gru)."""
    if kind == "masked":
        return MaskedReconPredictor(dim, hidden=hidden, **kwargs)
    if kind == "tcn":
        return TCNPredictor(dim, horizons=horizons, hidden=hidden)
    if kind == "gru":
        return GRUPredictor(dim, horizons=horizons, hidden=hidden)
    raise ValueError(f"Invalid predictor {kind}")


class MaskedReconPredictor(nn.Module):
    """Non-causal masked-reconstruction predictor (default part predictor).

    Masked positions are replaced by a learned mask token; a small
    transformer encoder mixes visible context from BOTH sides and emits a
    prediction per position. Output keeps the causal predictors' contract
    ``(B, H, D, T)`` with H=1 so scorers and losses are unchanged; the
    training loss counts masked positions via the collator mask.
    """

    def __init__(self, dim: int, hidden=None, nhead: int = 4,
                 num_layers: int = 2, dropout: bool = True):
        super().__init__()
        hidden = hidden or dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(
            dim, nhead, dim_feedforward=hidden,
            dropout=0.1 if dropout else 0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.horizons, self.dim = 1, dim
        nn.init.normal_(self.mask_token, std=0.02)

    @staticmethod
    def _sinusoidal_pe(tokens: int, dim: int, device, dtype):
        """Position table so identical windows at different offsets differ."""
        position = torch.arange(tokens, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype)
                        * (-math.log(10000.0) / dim))
        pe = torch.zeros(tokens, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return pe

    def forward(self, z: torch.Tensor, mask_pos=None) -> torch.Tensor:
        """Predict masked tokens; mask_pos (B, T) bool, True = masked.

        With mask_pos None every token stays visible (dense-loss mode).
        """
        b, d, t = z.shape
        seq = z.transpose(1, 2)
        if mask_pos is not None:
            m = mask_pos.to(torch.bool).unsqueeze(-1).expand(-1, -1, d)
            seq = torch.where(m, self.mask_token.expand(b, t, d), seq)
        seq = seq + self._sinusoidal_pe(t, d, z.device, z.dtype)
        out = self.norm(self.encoder(seq)).transpose(1, 2)  # (B, D, T)
        return out.unsqueeze(1)  # (B, 1, D, T): H=1 contract
