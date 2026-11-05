"""Preserve individual branch losses and couple future outputs with L1."""

import math

from losses.lewm_cross_attention import CrossAttentionLeWMLoss
from losses.reconstruction import ReconL1Loss


class RecCALoss(CrossAttentionLeWMLoss):
    def __init__(self, w_rec=1.0, w_similarity=1.0, **kwargs):
        super().__init__(**kwargs)
        if any(isinstance(w, bool) or not isinstance(w, (int, float))
               or not math.isfinite(w) or w < 0 for w in (w_rec, w_similarity)):
            raise ValueError("Rec/similarity weights must be finite and nonnegative")
        self.w_rec, self.w_similarity = w_rec, w_similarity
        self.rec_loss = ReconL1Loss()

    def forward(self, outputs):
        out = super().forward(outputs)
        rec = self.rec_loss({"recon": outputs["rec_full"], "target": outputs["rec_target"]})["loss"]
        if outputs["rec_future"].shape != outputs["recon"].shape:
            raise ValueError("Rec future and CA output must have identical shapes")
        similarity = (outputs["rec_future"] - outputs["recon"]).abs().mean()
        # No detach: agreement must train both heads.
        out["ca_pred_loss"] = out["pred_loss"]
        out["rec_loss"], out["similarity_loss"] = rec, similarity
        out["pred_loss"] = self.w_pred * out["pred_loss"] + self.w_rec * rec + self.w_similarity * similarity
        out["loss"] = out["loss"] + self.w_rec * rec + self.w_similarity * similarity
        return out
