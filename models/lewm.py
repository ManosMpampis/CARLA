"""Frozen LEWM features -> convolutional Q/K/V -> attention -> mirrored head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import encoders, actions, decoders, heads, engines, score_functions
from utils.reconstruction_scores import normalize_score_mode, reconstruction_score_map


class LeWM(nn.Module):
    """
    """

    def __init__(self, engine, backbone, encoder, action, decoder, head,
                 score_mode: str = "l1", score_fn: str = "reconstruction_input",
                 kwargs={}):
        super().__init__()
        freeze_backbone = backbone.pop("freeze", True)
        freeze_encoder = encoder.pop("freeze", False)
        freeze_action = action.pop("freeze", False)
        freeze_decoder = decoder.pop("freeze", False)
        freeze_head = head.pop("freeze", False)

        self.amp_enabled = kwargs.get("amp_enabled", False)
        self.score_mode = normalize_score_mode(score_mode)
        self.score_fn = score_functions(score_fn)

        self.action_strategy = action["strategy"]

        if self.action_strategy not in ("CEM", "trainable"):
            raise ValueError("action_strategy must be CEM or trainable")

        self.input_engine: nn.Module = engines(engine)

        self.backbone: nn.Module = encoders(backbone)
        self.encoder: nn.Module = encoders(encoder)
        self.action: nn.Module = actions(action)
        self.decoder: nn.Module = decoders(decoder)
        self.head: nn.Module = heads(head)

        if not isinstance(freeze_backbone, bool):
            raise ValueError("freeze_backbone must be true or false")
        self.backbone_frozen = freeze_encoder
        self.backbone.requires_grad_(not freeze_backbone)
        if freeze_backbone:
            self.backbone.eval()
        
        if not isinstance(freeze_encoder, bool):
            raise ValueError("freeze_encoder must be true or false")
        self.encoder_frozen = freeze_encoder
        self.encoder.requires_grad_(not freeze_encoder)
        if freeze_encoder:
            self.encoder.eval()

        if not isinstance(freeze_action, bool):
            raise ValueError("freeze_action must be true or false")
        self.action_frozen = freeze_action
        self.action.requires_grad_(not freeze_action)
        if freeze_action:
            self.action.eval()

        if not isinstance(freeze_decoder, bool):
            raise ValueError("freeze_decoder must be true or false")
        self.decoder_frozen = freeze_decoder
        self.decoder.requires_grad_(not freeze_decoder)
        if freeze_decoder:
            self.decoder.eval()

        if not isinstance(freeze_head, bool):
            raise ValueError("freeze_head must be true or false")
        self.freeze_head = freeze_decoder
        self.head.requires_grad_(not freeze_head)
        if freeze_head:
            self.head.eval()

    def forward(self, x):
        input = self.input_engine(x) # returns {"current_state": x_inj, "future_state": x, "query_start": 0, "target_range": (0, x.size(-1)), "mask": m_bool}
        
        z = self.backbone(input["future_state"])
        zhat = self.backbone(input["current_state"])
        z = self.encoder(z)
        zhat = self.encoder(zhat)

        if self.action_strategy == "trainable":
            a = self.action(zhat, z, input) # returns {latent representation that correcponds to the action, "mask_target": a float tensor like mask for auxilary heads to have a target to learn from}
        else:
            a = self.action(input) # returns {a latent representation that correcponds to known used actions of the world, "mask_target": a float tensor like mask for auxilary heads to have a target to learn from}
                    
        zhat_dot = self.predictor(zhat, a)
        out = self.head(zhat_dot)
        return {"prediction": zhat_dot,
                "latents_current_state": z,
                "latents_future_state": zhat,
                "output": out,
                "action": a}

    @torch.no_grad()
    def score(self, x):
        """Return only the target crop's score map, always in input timesteps."""
        was_training = self.training
        self.eval()
        try:
            with torch.autocast(device_type=x.device.type, enabled=self.amp_enabled):
                out = self(x)
                scores, _ = self.score_fn(out, self.score_mode)
                return {"scores": scores, "actions": out["action"], "latents": out["latents"], "action": out["action"], "encoder_output": out["encoder_output"]}
        finally:
            self.train(was_training)

    def latent_variance(self, latents):
        z = latents
        return z.transpose(1, 2).reshape(-1, z.size(1)).var(dim=0, unbiased=False).mean().item()

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        if self.encoder_frozen:
            self.encoder.eval()
        if self.action_frozen:
            self.action.eval()
        if self.decoder_frozen:
            self.decoder.eval()
        if self.freeze_head:
            self.freeze_head.eval()
        return self
    
    def update_ema(self):
        """No teacher: no-op satisfying the Trainer contract."""

    def update_running_stats(self, latents):
        """No running statistics on the bare trunk."""
