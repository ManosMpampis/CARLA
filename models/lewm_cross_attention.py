"""LEWM (Encoder) -> (Decoder) convolutional Q/V -> attention -> mirrored head.
                 |-> (Action)  convolution K ------>|
"""

import torch.nn as nn

from models.blocks import CrossAttention, StackedConv1dBlock, ResNetBlock, TCNEncoder, Encoder
from models.recon_head import build_mirrored_head
from utils.reconstruction_scores import reconstruction_score_map


class CrossAttentionPredictor(nn.Module):

    def __init__(self, encoder, action, kernels=(7,5,3), strides=(1, 1, 1), residual_kernel: int = 5,
                 norm: str = "batch", act: str = "relu", dropout: int = 0.1, attention_heads: int = 1,
                 input_projection=None, position_encoding="sinusoidal", position_on="both", diagonal_mask: bool = False):
        super().__init__()
        encoder_output = encoder.output_dims

        encoder_type = action.encoder_type
        action_params = {
            "in_dim": action.input_dims,
            "out_dims": action.output_dims,
            "hidden_dims": action.hiddent_dims,
            "kernels": action.kernels,
            "norm": action.norm,
            "act": action.act,
            "dropout": action.dropout,
        }
        assert encoder_output == action_params["in_dim"], "encoder and action should have the same output dim for cross entropy"
        self.attention_heads = attention_heads

        if encoder_type:
            self.convK = TCNEncoder(*action_params) # use the same encoder model
            self.convV = TCNEncoder(*action_params) # use the same encoder model
        else:
            self.convK = Encoder(*action_params)
            self.convV = Encoder(*action_params)

        self.cross_attention = CrossAttention(num_heads=attention_heads, dropout=dropout,
                                            input_projection=input_projection, position_encoding=position_encoding, 
                                            position_on=position_on, diagonal_mask=diagonal_mask)
        kernels = list(kernels)
        if len(kernels) == 0:
            self.pred = nn.Identity()
        else:
            strides = list(strides)
            residual_kernel = list(residual_kernel)
            if len(residual_kernel)==1:
                residual_kernel = residual_kernel * len(strides)

            if residual_kernel[0]>=0:
                self.pred = nn.Sequential(*
                    [
                        ResNetBlock(encoder_output, encoder_output, kernels=kernels, stride=st, residual_kernel=rk, norm=norm, act=act, dropout=dropout) for st, rk in zip(strides, residual_kernel)
                    ]
                )
            else:
                self.pred = StackedConv1dBlock(encoder_output, encoder_output, kernels=kernels, strides=strides, norm=norm, act=act, dropout=dropout)

    def forward(self, kv, q):
        # query should come from action
        key = self.convK(kv["key"])
        value = self.convV(kv["value"])

        x = self.cross_attention(key.clone(), q.clone(), value.clone())
        x = self.pred(x)
        return x
