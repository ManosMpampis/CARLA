"""Shared convolutional building blocks (configurable, documented).

All blocks are plain classes with methods. Kernel/stride/depth choices
come from config; scalar args broadcast over the block's convolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import Conv1dBlock, TCNEncoder, Encoder, _init_weights


class TimeAuxiliary(nn.Module):
    """Convolution time auxiliary: localizes mask, feeds FiLM features.

    Action is never an input here (LeWM requirement); the mask head learns
    purely from Encoder(X_inj).
    """

    def __init__(self, in_dim: int, aux_channels=(32, 32, 32),
                 kernels=(7, 5, 3), norm: str = "batch", act: str = "relu",
                 dropout: float = 0.1):
        super().__init__()
        ks = list(kernels)
        chs = [in_dim] + list(aux_channels)
        self.convs = nn.Sequential(
            *[Conv1dBlock(chs[i], chs[i + 1], kernel=ks[i], stride=1,
                          norm=norm, act=act, dropout=dropout)
              for i in range(len(chs) - 1)]
        )
        self.mask_head = nn.Conv1d(chs[-1], 1, kernel_size=1)
        _init_weights(self.mask_head)
        self.feat_dim = int(chs[-1])

    def forward(self, z):
        """Return (mask_logits (B, W), film_features (B, Da, W))."""
        h = self.convs(z)
        logits = self.mask_head(h).squeeze(1)
        return logits, h


class EncoderAction(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list, out_dims: int, kernels=(7, 5, 3), stride=(1, 1, 1),
                 norm: str = "batch", act: str = "relu", dropout: float = 0.1, encoder_type: str = "tcn"):
        super().__init__()
        self.encoder_type = encoder_type.lower()
        assert self.encoder_type in ("tcn", "normal"), "encoder_type should be tcn or normal"
        
        self.input_dims = in_dim
        self.hidden_dims = hidden_dims
        self.output_dims = out_dims
        self.kernels = kernels
        self.norm = norm
        self.act = act
        self.dropout = dropout
        
        if self.encoder_type == "tcn":
            self.net = TCNEncoder(in_dim, out_dims, kernels, norm, act, stride, dropout)
        else:
            self.net = Encoder(in_dim, out_dims, kernels, norm, act, stride, dropout)

    def forward(self, x):
        x = self.net(x)
        return {
            "action": {"action_token": x}
        }


class MaskedRepresentationAction(nn.Module):
    def __init__(self, encoder_input_dim):
        super().__init__()
        # Learned unknown-token: marks what-to-reconstruct
        # positions in the predictor input. When present, the auxiliary
        # still sees raw corrupted latents.
        self.mask_token = nn.Parameter(torch.zeros(1, encoder_input_dim, 1))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, zhat, z, input):
        """Blend learned token into masked positions (hard if binary).
        During training the m_float is the actual mask, during inference is the proposed mask from the auxiliary."""
        if "mask_float" not in input:
            raise ValueError("input must contain 'mask_float' key for MaskedRepresentationAction")
        m_float = input["mask_float"]
        if m_float.ndim == 1:
            m_float = m_float.unsqueeze(0)
        if m_float.size(-1) != z.size(-1):
            m_float = F.interpolate(m_float.unsqueeze(1).float(),
                                    size=z.size(-1),
                                    mode="nearest").squeeze(1).to(z.dtype)
        
        blend = m_float.unsqueeze(1).expand_as(zhat)
        tok = self.mask_token.expand_as(zhat)
        return {
            "action": {"action_token": blend * tok + (1 - blend) * zhat},
        }


class MaskedRepresentationAuxiliaryGuidedAction(nn.Module):
    def __init__(self, encoder_input_dim, aux_channels=32, aux_kernels=(3, 5, 7), norm="batch", dropout=0.0, aux_mask_during_training=True):
        super().__init__()
        # Learned unknown-token guided by auxilary head: marks what-to-reconstruct
        # positions in the predictor input. When present, the auxiliary
        # still sees raw corrupted latents.
        self.aux_mask_during_training = aux_mask_during_training
        self.mask_token = nn.Parameter(torch.zeros(1, encoder_input_dim, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        self.aux = TimeAuxiliary(encoder_input_dim, aux_channels=aux_channels,
                            kernels=tuple(aux_kernels), norm=norm,
                            dropout=dropout)

    def forward(self, zhat, z, input):
        """Blend learned token into masked positions (hard if binary).
        During training the m_float is the actual mask, during inference is the proposed mask from the auxiliary."""
        if "mask_float" not in input:
            raise ValueError("input must contain 'mask_float' key for MaskedRepresentationAuxiliaryGuidedAction")
        m_float = input["mask_float"]

        mask_logits, film_feat = self.aux(zhat)
        if not self.training or self.aux_mask_during_training:
            m_float = torch.sigmoid(mask_logits).detach().to(z.dtype) if mask_logits is not None \
                            else z.new_zeros(z.size(0), z.size(-1))
        if m_float.ndim == 1:
            m_float = m_float.unsqueeze(0)
        if m_float.size(-1) != z.size(-1):
            m_float = F.interpolate(m_float.unsqueeze(1).float(),
                                    size=z.size(-1),
                                    mode="nearest").squeeze(1).to(z.dtype)
        
        blend = m_float.unsqueeze(1).expand_as(zhat)
        tok = self.mask_token.expand_as(zhat)
        return {
            "action": {"action_token": blend * tok + (1 - blend) * zhat, "film_feat": film_feat},
        }
