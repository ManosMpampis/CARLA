"""Steered-LeWM contract check: ResNet encoder + time aux + freq predictor.

Asserts the spec mechanics without training on SMD:
- SubAnomalyMaskCollator emits matching (mask, X_inj); mask span is exact.
- SteeredFreqLeWMModel is single-scale L0/stride-1, no teacher, no detach.
- Auxiliary never takes the action (signature check); the predictor input
  is mask-token-blended (option 4): corrupted values never reach the
  predictor's direct path, while the auxiliary still sees them raw.
- SteeredLeWMLoss runs kl/mse/l1 + aux MSE + dual SIGReg with pred_loss key.
- Gradients reach encoder + aux + predictor + mask token;
  score() emits fused (B, W).
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
    model = SteeredFreqLeWMModel(
        built["model"], aux_channels=[8, 8, 8], stem_channels=8,
        neck_widths=[8, 8, 8], n_fft=32, hop_length=8, win_length=32)
    assert model.level_names == ["L0"] and model.level_strides == [1]
    return model


def main():
    torch.manual_seed(4)
    import numpy as np

    np.random.seed(4)
    model = _tiny_model()
    assert model.target_encoder is None and model.codebook is None
    assert model.mask_token.shape == (1, 12, 1)

    # Auxiliary never receives the action; predictor takes token-blended
    # latents plus steering features (no raw mask concat).
    assert list(inspect.signature(model.aux.forward).parameters) == ["z"]
    assert set(inspect.signature(model.predictor.forward).parameters) >= \
        {"z_tok", "aux_feat"}

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
    assert model.mask_token.grad is not None

    # Option 4 mechanism: blend puts the token exactly on masked spots.
    with torch.no_grad():
        z_probe = torch.randn(b, 12, w)
        m_probe = torch.zeros(b, w)
        m_probe[:, 10:30] = 1.0
        blended = model._token_blend(z_probe, m_probe)
        tok = model.mask_token.expand_as(z_probe)
        assert torch.equal(blended[:, :, 10:30], tok[:, :, 10:30])
        assert torch.equal(blended[:, :, :10], z_probe[:, :, :10])
        assert torch.equal(blended[:, :, 30:], z_probe[:, :, 30:])

    # End to end at init (FiLM is zero-init = identity): corrupted content
    # reaches the predictor only through steering features, never the direct
    # path. Same token input + different aux features -> identical output.
    # Eval mode silences dropout so the comparison is bitwise exact.
    model.eval()
    with torch.no_grad():
        z_inj1 = model.encode(mask["X_inj"])
        noisy_inj = mask["X_inj"] + 10.0 * torch.randn_like(mask["X_inj"])
        z_inj2 = model.encode(noisy_inj)
        zt = model.mask_token.expand_as(z_inj1)
        _, f1 = model.aux(z_inj1)
        _, f2 = model.aux(z_inj2)
        assert not torch.equal(f1, f2)  # steering sees the content difference
        p1 = model.predictor(zt, f1)
        p2 = model.predictor(zt, f2)
    assert torch.equal(p1, p2), "masked-span content leaked into predictor"

    # No-mask fallback (adapt path): dense clean reconstruction + empty mask.
    out_dense = model(x)
    assert out_dense["mask_target"].abs().sum() == 0
    assert torch.isfinite(crit(out_dense)["loss"])
    assert float((out["predicted"]["L0"] - out_dense["predicted"]["L0"])
                 .detach().abs().mean()) > 0

    model.eval()
    with torch.no_grad():
        s = model.score(x)
    assert s["fused"].shape == (b, w)
    assert s["levels"]["L0"].shape == (b, w)
    assert s["signals"] == {}

    # Eval-mode forward proposes the action (validation = inference path):
    # same input gives different predictions than teacher-forced train mode,
    # while the aux-localization target stays the given GT mask.
    model.train()
    with torch.no_grad():
        out_train = model(x, mask=mask)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, mask=mask)
    assert not torch.equal(out_train["predicted"]["L0"],
                           out_eval["predicted"]["L0"])
    assert torch.equal(out_eval["mask_target"], out_train["mask_target"])

    # Scoring batch size is plumbing only: identical fused scores and
    # identical window bookkeeping for tiny vs default batches.
    from utils.scoring import Scorer  # noqa: E402
    series = torch.randn(200, c).numpy().astype(np.float32)
    r_small = Scorer(model, torch.device("cpu"),
                     batch_size=2).score_series(series, 64, 7)
    r_big = Scorer(model, torch.device("cpu")).score_series(series, 64, 7)
    assert np.allclose(r_small["scores"], r_big["scores"],
                        rtol=1e-5, atol=1e-6)
    assert np.array_equal(r_small["start_idxs"], r_big["start_idxs"])
    assert np.array_equal(r_small["end_idxs"], r_big["end_idxs"])
    print("verify_steered_lewm: OK",
          {k: round(float(v.detach()), 4) for k, v in losses.items() if k != "loss"},
          "loss:", round(float(losses["loss"].detach()), 4))


if __name__ == "__main__":
    main()
