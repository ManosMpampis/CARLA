"""Action modules for mask-conditioned and guided latent prediction."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.blocks import Conv1dBlock, _init_weights, _per_layer


@dataclass
class ActionOutput:
    token_input: torch.Tensor | None = None
    film_feat: torch.Tensor | None = None
    key: torch.Tensor | None = None
    query: torch.Tensor | None = None
    mask_logits: torch.Tensor | None = None
    mask_target: torch.Tensor | None = None
    mask_proposal: torch.Tensor | None = None


class TimeAuxiliary(nn.Module):
    """Localize corruption and extract temporal features from one input view."""

    def __init__(self, in_dim, aux_channels=(32, 32, 32), kernels=(7, 5, 3),
                 norm="batch", act="gelu", dropout=0.1):
        super().__init__()
        channels = [in_dim, *aux_channels]
        kernels = _per_layer(kernels, len(aux_channels), "auxiliary kernels")
        self.convs = nn.Sequential(*[
            Conv1dBlock(a, b, kernel=k, norm=norm, act=act, dropout=dropout)
            for a, b, k in zip(channels, channels[1:], kernels)
        ])
        self.mask_head = nn.Conv1d(channels[-1], 1, kernel_size=1)
        _init_weights(self.mask_head)
        self.feat_dim = channels[-1]

    def forward(self, z):
        features = self.convs(z)
        return self.mask_head(features).squeeze(1), features


class MaskedRepresentationAction(nn.Module):
    """Replace masked latents with a learned token and supply FiLM features."""

    def __init__(self, latent_dim, aux=None, *, aux_input="latent", train_mask="predicted",
                 mask_reduction="max"):
        super().__init__()
        if aux_input not in ("latent", "raw"):
            raise ValueError("aux_input must be latent or raw")
        if train_mask not in ("known", "predicted"):
            raise ValueError("train_mask must be known or predicted")
        if train_mask == "predicted" and aux is None:
            raise ValueError("predicted masks require an auxiliary")
        if mask_reduction not in ("max", "nearest"):
            raise ValueError("mask_reduction must be max or nearest")
        self.aux = aux
        self.film_dim = aux.feat_dim if aux is not None else 0
        self.aux_input = aux_input
        self.train_mask = train_mask
        self.mask_reduction = mask_reduction
        self.mask_token = nn.Parameter(torch.empty(1, latent_dim, 1))
        nn.init.normal_(self.mask_token, std=0.02)

    def _token_blend(self, current, mask):
        weights = mask.unsqueeze(1).to(current.dtype)
        return current * (1 - weights) + self.mask_token.to(current.dtype) * weights

    def _token_mask(self, mask, tokens):
        if mask.size(-1) == tokens:
            return mask
        if self.mask_reduction == "nearest":
            return F.interpolate(mask.float().unsqueeze(1), size=tokens, mode="nearest").squeeze(1)
        return mask.reshape(mask.size(0), tokens, -1).amax(dim=-1)

    def forward(self, current, target, views, *, action=None, propose=False):
        tokens = current.size(-1)
        known = views.mask_target
        if known is None:
            known = current.new_zeros(current.size(0), views.current_input.size(-1))
        aux_input = views.current_input if self.aux_input == "raw" else current
        logits, features = self.aux(aux_input) if self.aux is not None else (None, None)

        if action is not None:
            if not torch.is_tensor(action) or action.shape not in (known.shape, (known.size(0), tokens)):
                raise ValueError("mask action must have shape (B, W) or (B, T_current)")
            mask = action.to(device=current.device)
        elif logits is not None and (propose or self.train_mask == "predicted"):
            mask = logits.sigmoid()
        else:
            mask = torch.zeros_like(known) if propose else known

        blend = self._token_mask(mask, tokens)
        if features is not None and features.size(-1) != tokens:
            features = F.adaptive_avg_pool1d(features, tokens)
        mask_target = known if self.aux_input == "raw" else self._token_mask(known, tokens)
        return ActionOutput(
            token_input=self._token_blend(current, blend),
            film_feat=features,
            mask_logits=logits,
            mask_target=mask_target,
            mask_proposal=blend,
        )


class EncoderAction(nn.Module):
    """Encode observed future patterns into length-preserving attention queries."""

    def __init__(self, encoder):
        super().__init__()
        if encoder.total_stride != 1:
            raise ValueError("guided action queries must preserve the target token count")
        self.net = encoder

    def forward(self, current, target, views, *, action=None, propose=False):
        if action is not None:
            raise ValueError("guided action queries are derived from the observed target")
        return ActionOutput(query=self.net(target))


class DiscreteAction(nn.Module):
    """Embed observed or planner-proposed action IDs as conditioning features."""

    def __init__(self, num_actions, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embedding_dim)
        self.film_dim = embedding_dim

    def forward(self, current, target, views, *, action=None, propose=False):
        if not torch.is_tensor(action) or action.dtype not in (torch.int32, torch.int64):
            raise ValueError("discrete action requires integer action IDs")
        if action.shape not in ((current.size(0),), (current.size(0), current.size(-1))):
            raise ValueError("action IDs must have shape (B,) or (B, T_current)")
        if action.ndim == 1:
            action = action.unsqueeze(-1)
        return ActionOutput(token_input=current, film_feat=self.embedding(action.to(current.device)).transpose(1, 2))


class NoAction(nn.Module):
    film_dim = 0

    def forward(self, current, target, views, *, action=None, propose=False):
        if action is not None:
            raise ValueError("this predictor has no action module")
        return ActionOutput(token_input=current)
