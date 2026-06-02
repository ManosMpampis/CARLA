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


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):

        kernel, dilation, stride = self.weight.size()[2], self.dilation[0], self.stride[0]
        l_out = l_in = input.size()[2]
        padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
        input = F.pad(input, [0, padding % 2])
        padding = (padding // 2,)
        output = F.conv1d(input=input, weight=self.weight, bias=self.bias, stride=stride,
                        padding=padding, dilation=dilation, groups=self.groups)
        return output


# def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
#     # stride and dilation are expected to be tuples.
#     kernel, dilation, stride = weight.size()[2], dilation[0], stride[0]
#     l_out = l_in = input.size()[2]
#     padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
#     input = F.pad(input, [0, padding % 2])
#     padding = (padding // 2,)
#     output = F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
#                     padding=padding,
#                     dilation=dilation, groups=groups)
#     return output

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, norm_layer_name: str ="batch", window_size: int = 0) -> None:
        super().__init__()
        if norm_layer_name == "layer":
            norm_layer = nn.LayerNorm([out_channels, window_size], elementwise_affine=True, bias=True)
        elif norm_layer_name == "instance":
            norm_layer = nn.InstanceNorm1d(num_features=out_channels, affine=True, bias=True)
        elif norm_layer_name in ["none", "no"]:
            norm_layer = nn.Identity()
        else:
            norm_layer = nn.BatchNorm1d(num_features=out_channels, affine=True, bias=True)

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            norm_layer,
        )
        
        # Initialize all weights and biases in this block
        for module in self.layers:
            _init_weights(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)