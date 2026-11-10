"""Compose view preparation, latent prediction and optional output reconstruction."""

from collections.abc import Callable
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.actions import DiscreteAction
from utils.reconstruction_scores import normalize_score_mode, reconstruction_score_map
from utils.model_contract import ModelContract


class LeWM(ModelContract, nn.Module):
    """One latent prediction pipeline with independent, swappable modules."""

    def __init__(self, *, engine, encoder, action, predictor, backbone=None, head=None,
                 freeze=(), score_domain="latent", score_mode="l1",
                 score_fn: Callable = reconstruction_score_map, legacy_optimizer_order=False,
                 score_tau=1.0):
        super().__init__()
        self.engine = engine
        self.backbone = backbone
        self.encoder = encoder
        self.action = action
        self.predictor = predictor
        self.head = head
        self.score_domain = score_domain
        self.score_mode = normalize_score_mode(score_mode)
        self.score_fn = score_fn
        self.score_tau = float(score_tau)
        if not math.isfinite(self.score_tau) or self.score_tau <= 0:
            raise ValueError("score_tau must be finite and positive")
        self._legacy_optimizer_order = legacy_optimizer_order
        if score_domain not in ("latent", "raw") or (score_domain == "raw" and head is None):
            raise ValueError("score_domain must be latent or raw with a reconstruction head")

        self.total_stride = encoder.total_stride * getattr(backbone, "total_stride", 1)
        self.level_names = ["L0"]
        self.level_dims = [encoder.output_dims]
        self.level_strides = [self.total_stride]
        self.score_input_resolution = True
        self.target_encoder = None
        self.codebook = None
        self.anti_collapse = "sigreg"
        self.encoder_frozen = False
        self._frozen = set()
        self.freeze(*freeze)

    def freeze(self, *roles):
        for role in roles:
            module = getattr(self, role)
            if module is None:
                continue
            module.requires_grad_(False).eval()
            if role == "encoder":
                self.encoder_frozen = True
            else:
                self._frozen.add(role)

    @property
    def aux(self):
        return getattr(self.action, "aux", None)

    @property
    def mask_token(self):
        return self.action.mask_token

    def _token_blend(self, current, mask):
        return self.action._token_blend(current, mask)

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        for name in self._frozen:
            getattr(self, name).eval()
        return self

    def encode(self, x):
        if x.size(-1) < self.total_stride or x.size(-1) % self.total_stride:
            raise ValueError("input length must be a positive multiple of the feature stride")
        return self.encoder(self.backbone(x) if self.backbone is not None else x)

    def predict_action(self, current_latent, action_ids):
        """Roll out one transition using a caller-supplied finite action."""
        if not isinstance(self.action, DiscreteAction):
            raise ValueError("predict_action requires the discrete action module")
        condition = self.action(current_latent, None, None, action=action_ids)
        return self.predictor.predict(current_latent, condition, target_tokens=current_latent.size(-1))

    def target_input_range(self, window):
        return self.engine.target_input_range(window)

    def forward(self, x, mask=None, action=None, *, corrupt=True):
        views = self.engine(x, mask, corrupt=corrupt and action is None)
        current = self.encode(views.current_input)
        target = current if views.target_input is views.current_input else self.encode(views.target_input)
        condition = self.action(current, target, views, action=action, propose=not self.training)
        query_start = (views.target_range[0] - views.current_range[1]) // self.total_stride
        prediction = self.predictor.predict(current, condition, target_tokens=target.size(-1),
                                            query_start=query_start)
        if prediction.shape != target.shape:
            raise ValueError(f"predicted latent {prediction.shape} must match target {target.shape}")
        out = {"context": {"L0": current}, "targets": {"L0": target},
               "latents": {"L0": target}, "predicted": {"L0": prediction},
               "mask": None if views.mask_target is None else {"input": views.mask_target},
               "mask_target": condition.mask_target, "mask_logits": condition.mask_logits,
               "action": condition, "target_range": views.target_range}
        if self.head is not None:
            recon = self.head(prediction)
            if recon.shape != views.target_input.shape:
                raise ValueError(f"head output {recon.shape} must match raw target {views.target_input.shape}")
            out.update(recon=recon, target=views.target_input)
        return out

    @torch.no_grad()
    def score(self, x, action=None):
        was_training = self.training
        self.eval()
        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                out = self(x.float(), action=action, corrupt=False)
                if self.score_domain == "raw":
                    fused, _ = self.score_fn(out["recon"], out["target"], self.score_mode,
                                             tau=self.score_tau)
                else:
                    fused, _ = self.score_fn(out["predicted"]["L0"], out["targets"]["L0"],
                                             self.score_mode, tau=self.score_tau)
                    raw_length = out["target_range"][1] - out["target_range"][0]
                    if fused.size(-1) != raw_length:
                        fused = F.interpolate(fused.unsqueeze(1), size=raw_length,
                                              mode="linear", align_corners=False).squeeze(1)
                if fused.size(-1) != out["target_range"][1] - out["target_range"][0]:
                    raise ValueError("score length must match target span")
                return {"fused": fused, "levels": {"L0": fused}, "signals": {}}
        finally:
            self.train(was_training)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Old pretraining checkpoints stored auxiliary/token weights at the root.
        token = prefix + "mask_token"
        if token in state_dict:
            state_dict[prefix + "action.mask_token"] = state_dict.pop(token)
            aux_prefix = prefix + "aux."
            for key in list(state_dict):
                if key.startswith(aux_prefix):
                    state_dict[prefix + "action." + key.removeprefix(prefix)] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def optimizer_parameters(self):
        """Keep the historical parameter order for masked LeWM checkpoints."""
        if self._legacy_optimizer_order:
            yield self.action.mask_token
            yield from self.encoder.parameters()
            if self.aux is not None:
                yield from self.aux.parameters()
            yield from self.predictor.parameters()
        else:
            yield from self.parameters()

    def update_ema(self):
        """LeWM uses a shared encoder."""

    def update_running_stats(self, latents):
        """LeWM has no extra running statistics."""
