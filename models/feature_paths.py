"""Compose trained feature paths while retaining decoder geometry."""

from torch import nn


class FeaturePath(nn.Sequential):
    def __init__(self, modules):
        super().__init__(*modules)
        if not modules:
            raise ValueError("feature path must contain a module")
        self.input_dims = modules[0].input_dims
        self.output_dims = modules[-1].output_dims
        self.total_stride = 1
        self.channel_ladder = [self.input_dims]
        self.enc_strides = []
        for module in modules:
            if module.input_dims != self.channel_ladder[-1]:
                raise ValueError("feature path channel widths must agree")
            self.channel_ladder.extend(module.channel_ladder[1:])
            self.enc_strides.extend(module.enc_strides)
            self.total_stride *= module.total_stride
