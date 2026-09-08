"""Steered-LeWM contract check: ResNet encoder + time aux + freq predictor.

Asserts the spec mechanics without training on SMD:
- SubAnomalyMaskCollator emits matching (mask, X_inj); mask span is exact.
- SteeredFreqLeWMModel is single-scale L0/stride-1, no teacher, no detach.
- Auxiliary never takes the action (signature check); mask feeds only the
  frequency stem.
- SteeredLeWMLoss runs kl/mse/l1 + aux MSE + dual SIGReg with pred_loss key.
- Gradients reach encoder + aux + predictor; score() emits fused (B, W) and
  the self-proposed-action path differs from the zero-mask path.
"""
import inspect
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from losses.steered_lewm import SteeredLeWMLoss  # noqa: E402
from models import get_backbone  # noqa: E402
from models.steered_lewm import SteeredFreqLeWMModel  # noqa: E402
from utils.masking_steered import SubAnomalyMaskCollator  # noqa: E402


def _tiny_model():
    built = get_backbone("steered_resnet", in_channels=4,
                         enc_channels=[8, 12], dropout=False)
    assert built["model"].level_names == ["L0"]
    assert built["model"].level_strides == [1]
    return SteeredFreqLeWMModel(
        built["model"], aux_channels=[8, 8, 8], stem_channels=8,
        neck_widths=[8, 8, 8], n_fft=32, hop_length=8, win_length=32)


def main():
    torch.manual_seed(4)
    import numpy as np

    np.random.seed(4)
    model = _tiny_model()
    assert model.target_encoder is None and model.codebook is None
    assert model.level_names == ["L0"] and model.level_strides == [1]

    # Auxiliary never receives the action.
    sig = inspect.signature(model.aux.forward)
    assert list(sig.parameters) == ["z"], sig.parameters
    psig = inspect.signature(model.predictor.forward)
    assert set(psig.parameters) >= {"z_inj", "mask", "aux_feat"}

    b, w, c = 4, 64, 4
    x = torch.randn(b, c, w)
    col = SubAnomalyMaskCollator(min_ratio=0.1, max_ratio=0.9)
    mask = col(b, w, model.level_strides, x)
    assert mask["input"].shape == (b, w)
    assert mask["X_inj"].shape == (b, c, w)
    assert mask["input"].any() and (~mask["input"]).any()
    # Injection actually changes values inside the mask span.
    diff = (mask["X_inj"] - x).abs()
    inside = diff[mask["input"].unsqueeze(1).expand_as(diff)].mean()
    assert float(inside) > 0, "injection left masked span unchanged"

    out = model(x, mask=mask)
    assert set(out) >= {"context", "latents", "targets", "predicted",
                        "mask_logits", "mask_target", "mask", "action"}
    assert out["latents"]["L0"].shape == (b, 12, w)
    assert out["predicted"]["L0"].shape == (b, 12, w)
    assert out["mask_logits"].shape == (b, w)

    for kind in ("kl", "mse", "l1"):
        crit = SteeredLeWMLoss(loss2_kind=kind, sigreg_kwargs={"num_slices": 4})
        losses = crit(out)
        assert {"loss", "pred_loss", "aux_loss", "sigreg", "sigreg_tgt"} <= set(losses)
        assert torch.isfinite(losses["loss"]), kind

    crit = SteeredLeWMLoss(loss2_kind="kl", sigreg_kwargs={"num_slices": 4})
    losses = crit(out)
    losses["loss"].backward()
    assert all(p.grad is not None for p in model.encoder.parameters()
               if p.requires_grad)
    assert all(p.grad is not None for p in model.aux.parameters()
               if p.requires_grad)
    assert any(p.grad is not None for p in model.predictor.parameters()
               if p.requires_grad)

    # No-mask fallback (adapt path): dense clean reconstruction + empty mask.
    out_dense = model(x)
    assert out_dense["mask_target"].abs().sum() == 0
    assert torch.isfinite(crit(out_dense)["loss"])

    model.eval()
    with torch.no_grad():
        s = model.score(x)
    assert s["fused"].shape == (b, w)
    assert s["levels"]["L0"].shape == (b, w)
    assert s["signals"] == {}
    # Self-proposed action matters: zero-masked prediction differs.
    with torch.no_grad():
        z = model.encode(x)["L0"]
        logits, feat = model.aux(z)
        zp_hat = model.predictor(z, torch.sigmoid(logits), feat)
        zp_zero = model.predictor(z, torch.zeros_like(torch.sigmoid(logits)), feat)
    assert float((zp_hat - zp_zero).abs().mean()) > 0
    print("verify_steered_lewm: OK",
          {k: round(float(v.detach()), 4) for k, v in losses.items() if k != "loss"},
          "loss:", round(float(losses["loss"].detach()), 4))


if __name__ == "__main__":
    main()
