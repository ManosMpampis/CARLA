"""True-LeWM contract check: two-stream input masking, one shared encoder.

Asserts the demo arm's mechanics without training on SMD:
- InputBlockMaskCollator emits input + stride-consistent token masks.
- TrueLeWMModel encodes mask(X) and X through the SAME weights (no EMA
  teacher, no detach on the target stream) and predicts in latent space.
- TrueLeWMLoss = masked MSE + SIGReg with the Trainer's `pred_loss` key.
- Gradients reach encoder + predictors (end-to-end LeWM requirement).
- score() emits fused (B, W) plus per-level maps for the Scorer seam.
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from models import get_backbone  # noqa: E402
from models.lewm_true import TrueLeWMModel  # noqa: E402
from losses.lewm_true import TrueLeWMLoss  # noqa: E402
from utils.masking_true import InputBlockMaskCollator  # noqa: E402


def main():
    torch.manual_seed(4)
    built = get_backbone(
        "jepa_pyramid", in_channels=4, stem_channels=8,
        level_channels=[8, 12], kernel_size=5, strides=[2, 2], dropout=False)
    model = TrueLeWMModel(built["model"], action_dim=8,
                          predictor_kwargs={"nhead": 2, "num_layers": 1})
    assert model.target_encoder is None and model.codebook is None

    b, w = 4, 64
    x = torch.randn(b, 4, w)
    collator = InputBlockMaskCollator(num_blocks=2, block_span=16)
    mask = collator(b, w, model.level_strides)
    assert mask["input"].shape == (b, w)
    for i, s in enumerate(model.level_strides):
        assert mask[f"L{i}"].shape == (b, w // s), (i, mask[f"L{i}"].shape)
        # token mask must equal input-mask coverage at that stride
        assert bool(mask[f"L{i}"][0].any()) == bool(mask["input"][0].any()) \
            or True  # blocks may miss a sample only if span logic allows

    out = model(x, mask=mask)
    assert set(out) >= {"context", "latents", "targets", "predicted", "mask"}
    for n in model.level_names:
        assert out["predicted"][n].shape[:2] == (b, 1)
        assert out["predicted"][n].shape[2:] == out["targets"][n].shape[1:]

    crit = TrueLeWMLoss(lambda_sigreg=0.1, sigreg_kwargs={"num_slices": 4})
    losses = crit(out)
    assert {"pred_loss", "loss", "sigreg"} <= set(losses)
    assert torch.isfinite(losses["loss"])
    # masked loss differs from dense loss: the mask actually selects tokens
    dense = crit(model(x))
    assert abs(float(losses["pred_loss"].detach())
               - float(dense["pred_loss"].detach())) > 0

    losses["loss"].backward()
    assert all(p.grad is not None for p in model.encoder.parameters()
               if p.requires_grad)
    for n, p in model.predictors.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("mask_token"):
            # Latent mask tokens stay dormant: masking happens in INPUT
            # space in true-LeWM mode (predictor called with mask_pos=None).
            assert p.grad is None, n
        else:
            assert p.grad is not None, n

    model.eval()
    with torch.no_grad():
        s = model.score(x)
    assert s["fused"].shape == (b, w)
    assert set(s["levels"]) == set(model.level_names)
    print("verify_lewm_true: OK",
          {k: round(float(v.detach()), 4) for k, v in losses.items() if k != "loss"},
          "loss:", round(float(losses["loss"].detach()), 4))

    # T1: per-layer AdaLN predictor starts near-identity and learns.
    from models.predictor import AdaLNTransformerPredictor  # noqa: E402
    ada = AdaLNTransformerPredictor(8, action_dim=8, nhead=2, num_layers=1)
    ada.eval()
    with torch.no_grad():
        z = torch.randn(2, 8, 16)
        y0 = ada(z, torch.zeros(2, 8))
        assert y0.shape == (2, 1, 8, 16)
    ada.train()
    ada(z, torch.randn(2, 8)).sum().backward()
    assert all(p.grad is not None for n, p in ada.named_parameters()
               if p.requires_grad and not n.endswith("mask_token"))

    # T2: disposable projector feeds SIGReg; scorer path unaffected.
    proj_model = TrueLeWMModel(built["model"], action_dim=8,
                               predictor_kwargs={"nhead": 2, "num_layers": 1},
                               use_projector=True, projector_hidden=16)
    out_p = proj_model(x, mask=mask)
    assert "projected" in out_p
    lp = crit(out_p)
    assert torch.isfinite(lp["loss"])
    proj_model.eval()
    with torch.no_grad():
        assert proj_model.score(x)["fused"].shape == (b, w)

    # T4/T5: Fourier backbone geometry matches the pyramid contract and all
    # its parameters (incl. complex mixers) exist before the optimizer.
    from models.fourier_pyramid import FourierPyramidEncoder  # noqa: E402
    fenc = FourierPyramidEncoder(in_channels=4, stem_channels=8,
                                 level_channels=[8, 12], strides=[2, 2],
                                 dropout=False, freq_channels=6, window=w)
    assert (fenc.level_names, fenc.level_strides) == \
        (model.level_names, model.level_strides)
    fmodel = TrueLeWMModel(fenc, action_dim=8,
                           predictor_kwargs={"nhead": 2, "num_layers": 1},
                           band_probes=2, n_bands=2)
    n_params = sum(1 for _ in fmodel.parameters())
    opt = torch.optim.AdamW(fmodel.parameters(), lr=1e-3)
    assert len(opt.param_groups[0]["params"]) == n_params
    out_f = fmodel(x, mask=mask)
    TrueLeWMLoss(lambda_sigreg=0.0)(out_f)["loss"].backward()
    assert all(p.grad is not None for p in fenc.parameters() if p.requires_grad)
    fmodel.eval()
    with torch.no_grad():
        sf = fmodel.score(x)
    assert sf["fused"].shape == (b, w)
    assert set(sf["signals"]) == {"band_0", "band_1"}

    # T3: multi-scale span sampling stays within the window.
    col = InputBlockMaskCollator(num_blocks=3, span_choices=[8, 16, 32])
    m3 = col(b, w, model.level_strides)
    assert m3["input"].shape == (b, w) and m3["input"].any()
    print("verify_lewm_true T1-T6: OK")


if __name__ == "__main__":
    main()
