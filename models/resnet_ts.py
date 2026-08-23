import torch
from torch import nn
import torch.nn.functional as F
from typing import List

from models.convolutions import ConvBlock, _init_weights


class ResNetBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_sizes: List[int] = [8, 5, 3],
            norm_layer_name: str = "batch",
            window_size: int = 0,
            dropout: bool = True,
    ) -> None:
        super().__init__()

        channels = [in_channels] + [out_channels for _ in range(len(kernel_sizes))]
        assert len(kernel_sizes) >= 2

        self.layers = nn.Sequential(
            *[
                layer
                for i in range(len(channels) - 1)
                for layer in (
                    ConvBlock(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_layer_name=norm_layer_name,
                        window_size=window_size,
                        dropout=dropout,
                    ),
                    nn.ReLU(),
                )
            ][:-1]
        )
        residual = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                norm_layer_name=norm_layer_name,
                window_size=window_size,
                dropout=dropout,
            )
        self.residual = nn.Identity() if in_channels == out_channels else residual
        self.act = nn.ReLU()
        
        # Initialize all weights and biases in this block
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all layers in this block."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d, 
                                  nn.InstanceNorm1d, nn.LayerNorm)):
                _init_weights(module)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        # for layer in self.layers:
        x = self.layers(x)

        x = self.residual(input) + x
        x = self.act(x)
        return x

class ResNetRepresentation(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The output of each residual block
    kernel_sizes:
        The kernel size of each convolution inside the residual block
    """

    def __init__(self, in_channels: int, mid_channels: List[int] = [4, 8, 8], kernel_sizes: List[int]|List[List[int]] = [8, 5, 3], norm_layer_name: str = "batch", window_size: int = 0, dropout: bool = True) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
        }

        kernel_sizes = kernel_sizes if isinstance(kernel_sizes[0], List) else [kernel_sizes for _ in range(len(mid_channels))]
        assert len(kernel_sizes) == len(mid_channels)

        channels = [in_channels] + mid_channels

        self.layers = nn.Sequential(
            *[
                ResNetBlock(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    norm_layer_name=norm_layer_name,
                    window_size=window_size,
                    dropout=dropout,
                    ) for i in range(len(mid_channels))
                ]
            )
        
        # Initialize all weights and biases in this representation
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all layers in this representation."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d, 
                                  nn.InstanceNorm1d, nn.LayerNorm)):
                _init_weights(module)

    def forward(self, x: torch.Tensor):
        z = self.layers(x)
        return z

def resnet_ts(**kwargs):
    return {'backbone': ResNetRepresentation(**kwargs), 'dim': kwargs['mid_channels']}#*2}