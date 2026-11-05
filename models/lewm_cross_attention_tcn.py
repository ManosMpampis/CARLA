"""Cross-attention variant with causal temporal extraction of observed queries.

Q comes from the second crop; K/V retain the original context convolutions.
Causality applies to feature tokens entering the TCN, not to the preceding
ResNet encoder or to the reconstruction head. This is conditional
reconstruction, not AxonAD's shifted, past-only query forecasting objective.
"""

import torch.nn as nn
import torch.nn.functional as F

from models.blocks import _init_weights
from models.lewm_cross_attention import CrossAttentionLeWM


class CausalConv1d(nn.Module):
    """Length-preserving convolution using the current and earlier tokens."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class CausalQueryTCN(nn.Module):
    """AxonAD-style dilated Conv/ReLU/dropout stack on [B,C,T] features."""

    def __init__(self, in_channels, out_channels, hidden_channels=None,
                 dilations=(1, 2, 4, 8), kernel_size=3, dropout=0.1):
        super().__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        widths = (in_channels, out_channels, hidden_channels, kernel_size)
        if any(type(value) is not int or value < 1 for value in widths):
            raise ValueError("TCN widths and kernel_size must be positive integers")
        if (not isinstance(dilations, (tuple, list)) or not dilations
                or any(type(value) is not int or value < 1 for value in dilations)):
            raise ValueError("TCN dilations must be a nonempty list of positive integers")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("TCN dropout must be in [0, 1)")
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        layers = []
        width = in_channels
        for dilation in dilations:
            layers.extend((CausalConv1d(width, hidden_channels, kernel_size, dilation),
                           nn.ReLU(), nn.Dropout(dropout)))
            width = hidden_channels
        layers.append(nn.Conv1d(hidden_channels, out_channels, kernel_size=1))
        self.net = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, x):
        return self.net(x)


class CrossAttentionTCNLeWM(CrossAttentionLeWM):
    """Separate model arm sharing crop, attention, loss and scoring contracts."""

    def __init__(self, encoder, *, query_tcn_hidden_channels=None,
                 query_tcn_dilations=(1, 2, 4, 8), query_tcn_kernel_size=3,
                 query_tcn_dropout=0.1, **kwargs):
        if kwargs.get("query_source", "observed") != "observed":
            raise ValueError("causal query TCN requires query_source=observed (second crop)")
        super().__init__(encoder, **kwargs)
        self.qkv_encoder.query = CausalQueryTCN(
            encoder.output_dims, self._task["qk_channels"],
            hidden_channels=query_tcn_hidden_channels,
            dilations=query_tcn_dilations, kernel_size=query_tcn_kernel_size,
            dropout=query_tcn_dropout)
        self._task.update(
            model_variant="causal_tcn_query",
            query_tcn_hidden_channels=(encoder.output_dims if query_tcn_hidden_channels is None
                                       else query_tcn_hidden_channels),
            query_tcn_dilations=list(query_tcn_dilations),
            query_tcn_kernel_size=query_tcn_kernel_size,
            query_tcn_dropout=float(query_tcn_dropout))
