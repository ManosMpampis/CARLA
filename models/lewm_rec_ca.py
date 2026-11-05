"""Joint full-window latent reconstruction and observed-query cross attention."""

import torch

from models.lewm_cross_attention_tcn import CrossAttentionTCNLeWM
from models.recon_head import build_mirrored_head
from utils.reconstruction_scores import reconstruction_score_map


class RecCALeWM(CrossAttentionTCNLeWM):
    """Shared frozen phase-one encoder; independent Rec/CA heads.

    Crops use raw input timesteps. Only history enters K/V. Future features
    enter Q, so this is conditional prediction, not past-only forecasting.
    The query TCN preserves token length; the pretrained encoder is noncausal.
    """

    def __init__(self, encoder, *, history_length=256, future_length=128,
                 **kwargs):
        if any(type(n) is not int or n < 1 or n % encoder.total_stride
               for n in (history_length, future_length)):
            raise ValueError("history/future lengths must be positive multiples of encoder stride")
        for key, required in dict(target="features", query_source="observed",
                                  feature_extraction="separate", freeze_encoder=True).items():
            if kwargs.pop(key, required) != required:
                raise ValueError(f"Rec+CA requires {key}={required}")
        if kwargs.pop("context_crop", None) is not None or kwargs.pop("target_crop", None) is not None:
            raise ValueError("Rec+CA uses history_length/future_length in raw timesteps")
        width = encoder.output_dims
        super().__init__(encoder, target="features", query_source="observed",
                         feature_extraction="separate", freeze_encoder=True,
                         **kwargs)
        self.history_length, self.future_length = history_length, future_length
        self.rec_head = build_mirrored_head(
            encoder, out_channels=width, upsample=False,
            norm=kwargs.get("norm", "batch"), dropout=kwargs.get("dropout", 0.0))
        self._task.update(model_variant="rec_ca_query_tcn", history_length=history_length,
                          future_length=future_length)

    def crop_ranges(self, window):
        if self.context_crop is not None or self.target_crop is not None:
            raise ValueError("Rec+CA evaluation uses configured history_length/future_length")
        if window != self.history_length + self.future_length:
            raise ValueError("Rec+CA window must equal history_length + future_length")
        boundary = self.history_length // self.total_stride
        return (0, boundary), (boundary, window // self.total_stride)

    def forward(self, x, mask=None, action=None):
        # CA separately encodes history/future to isolate history receptive fields.
        out = super().forward(x, mask=mask, action=action)
        # Match the standalone reconstruction arm: clean, full-window input.
        with torch.no_grad():
            full_target = self.encoder(x)
        rec = self.rec_head(full_target)
        _, (a, b) = self.crop_ranges(x.size(-1))
        out.update(rec_full=rec, rec_target=full_target, rec_future=rec[..., a:b])
        return out

    @torch.no_grad()
    def score(self, x):
        """Mean of Rec and CA residuals on the future span; fp32 only."""
        was_training = self.training
        self.eval()
        try:
            with torch.autocast(device_type=x.device.type, enabled=False):
                out = self(x.float())
                _, (a, b) = self.crop_ranges(x.size(-1))
                rec, _ = reconstruction_score_map(out["rec_future"], out["rec_target"][..., a:b], self.score_mode)
                ca, _ = reconstruction_score_map(out["recon"], out["target"], self.score_mode)
                rec, ca = [s.repeat_interleave(self.total_stride, -1) for s in (rec, ca)]
                return {"fused": (rec + ca) / 2, "levels": {"L0": (rec + ca) / 2},
                        "signals": {"rec": rec, "ca": ca}}
        finally:
            self.train(was_training)
