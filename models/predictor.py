"""Masked-part predictors (causal) + conditional wrapper for action arm.

After forecasting was dropped, `horizons` means part offsets: entry
[:, k-1, :, t] predicts the latent at position t+k from positions <= t.
Causality is kept so a part never sees itself.
"""
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

    def forward(self, z):
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

    def forward(self, z):
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

    def forward(self, z, action=None):
        """Predict with optional action; None disables conditioning."""
        if action is None:
            return self.base(z)
        # Broadcast per-window scale/shift over time.
        scale = 1 + self.to_scale(action).unsqueeze(-1)
        shift = self.to_shift(action).unsqueeze(-1)
        return self.base(z * scale + shift)


def build_predictor(kind: str, dim: int, horizons: int, hidden=None):
    """Factory honoring the predictor registry names (tcn|gru)."""
    if kind == "tcn":
        return TCNPredictor(dim, horizons=horizons, hidden=hidden)
    if kind == "gru":
        return GRUPredictor(dim, horizons=horizons, hidden=hidden)
    raise ValueError(f"Invalid predictor {kind}")
