"""LEWM prediction loss and dual SIGReg on the trainable Q/K/V encoder."""

from losses.lewm import LeWMLoss


class CrossAttentionLeWMLoss(LeWMLoss):
    """No target detach: the Query target remains a trainable LEWM branch.

    Features/raw targets are fixed because the feature extractor is frozen.
    Both clean and injected Q/K/V streams receive Gaussian regularization;
    the attention+head predictor is supervised in the chosen output domain.
    """

    def __init__(self, loss2_kind="l1", w_pred=1.0, lambda_sigreg=0.1,
                 lambda_sigreg_tgt=0.1, tau=1.0, sigreg_kwargs=None):
        super().__init__(loss2_kind=loss2_kind, w_pred=w_pred, w_aux=0.0,
                         lambda_sigreg=lambda_sigreg,
                         lambda_sigreg_tgt=lambda_sigreg_tgt, tau=tau,
                         sigreg_kwargs=sigreg_kwargs)

    def forward(self, outputs):
        pred, target = outputs["recon"], outputs["target"]
        if pred.shape != target.shape:
            raise ValueError(f"prediction {pred.shape} and target {target.shape} differ")
        pred_loss = self._latent_loss(pred, target)
        sig = self.sigreg(outputs["qkv"]) if self.lambda_sigreg else pred.new_zeros(())
        sig_tgt = self.sigreg(outputs["clean_qkv"]) if self.lambda_sigreg_tgt else pred.new_zeros(())
        return {"loss": self.w_pred * pred_loss + self.lambda_sigreg * sig
                        + self.lambda_sigreg_tgt * sig_tgt,
                "pred_loss": pred_loss, "sigreg": sig, "sigreg_tgt": sig_tgt}
