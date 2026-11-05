"""Frozen LEWM features -> convolutional Q/K/V -> attention -> mirrored head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import encoders, actions, decoders, engines, score_functions
from utils.reconstruction_scores import normalize_score_mode, reconstruction_score_map


class LeWM(nn.Module):
    """Two crop LEWM with a fixed feature extractor and trainable Q/K/V encoder.

    Crop intervals are half-open FEATURE TOKEN indices. With stride S,
    [a,b] refers to raw input [a*S,b*S]. Omitted intervals split in half.
    ``observed`` obtains Query from the second crop (conditional reconstruction).
    ``context`` obtains Query from the first crop (past-only forecasting).
    Separate extraction isolates receptive fields; full extraction is available
    for the observed arm to literally crop one full-window feature map.
    """

    def __init__(self, engine, backbone, encoder, action, decoder, score_mode: str = "reconstruction_input",
                 freeze_backbone: bool = True, freeze_encoder: bool = True, freeze_action: bool = True, freeze_decoder: bool = True,
                 kwargs={}):
        super().__init__()
        self.amp_enabled = kwargs.get("amp_enabled", False)
        self.score_mode = normalize_score_mode(score_mode)

        self.action_strategy = action["strategy"]

        if self.action_strategy not in ("CEM", "trainable"):
            raise ValueError("action_strategy must be CEM or trainable")

        self.input_engine: nn.Module = engines(engine)
        self.backbone: nn.Module = encoders(backbone)
        self.encoder: nn.Module = encoders(encoder)
        self.action: nn.Module = actions(action)
        self.decoder: nn.Module = decoders(decoder)
        self.score_fn = score_functions(score_mode)

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
        return {"prediction": zhat_dot,
                "latents_current_state": z,
                "latents_future_state": zhat,
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

    def update_ema(self):
        """No teacher: no-op satisfying the Trainer contract."""

    def update_running_stats(self, latents):
        """No running statistics on the bare trunk."""
