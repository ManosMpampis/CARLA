import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda")

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
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)