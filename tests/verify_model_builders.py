"""Model construction contracts: automatic decoder mirroring and bottlenecks."""

from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.builders import (get_ae_model, get_cross_attention_model, get_lewm_model,
                             get_recon_model, get_vae_model)
from models.recon_head import build_mirrored_head


def verify_mirror():
    config = {"model_kwargs": {"in_channels": 4, "enc_channels": [8, 12],
                               "enc_strides": [2, 2], "dropout": 0.0},
              "cross_attention_kwargs": {"target": "input", "qk_channels": 4,
                                         "value_channels": 6, "num_heads": 2,
                                         "context_crop": [0, 8], "target_crop": [8, 16]}}
    model = get_cross_attention_model(config)
    head = model.predictor.head
    reference = build_mirrored_head(model.encoder)
    actual = [(block.deconv.in_channels, block.deconv.out_channels,
               block.deconv.kernel_size, block.deconv.stride)
              for block in head.blocks]
    expected = [(block.deconv.in_channels, block.deconv.out_channels,
                 block.deconv.kernel_size, block.deconv.stride)
                for block in reference.blocks]
    assert actual == expected, f"decoder does not mirror encoder: {actual} != {expected}"
    assert head.to_input.in_channels == reference.to_input.in_channels
    assert head.to_input.out_channels == reference.to_input.out_channels
    assert model.predictor.input_projection.in_channels == 6
    assert model.predictor.input_projection.out_channels == 12
    assert model.encoder_frozen
    assert all(not param.requires_grad for param in model.encoder.parameters())
    assert model(torch.randn(2, 4, 64))["recon"].shape == (2, 4, 32)
    print("Encoder mirror preserved behind a narrow attention bottleneck: OK")


def verify_targets_and_freezing():
    for channels, strides in (([8, 12], [1, 1]), ([8, 12], [2, 2]),
                              ([8, 12, 16], [1, 2, 2])):
        for target in ("features", "input", "query"):
            config = {"model_kwargs": {"in_channels": 4, "enc_channels": channels,
                                       "enc_strides": strides, "dropout": 0.0},
                      "cross_attention_kwargs": {"target": target, "qk_channels": 4,
                                                 "value_channels": 6, "num_heads": 2}}
            model = get_cross_attention_model(config)
            expected = build_mirrored_head(model.encoder)
            actual = model.predictor.head
            assert [(b.deconv.in_channels, b.deconv.out_channels) for b in actual.blocks] == [
                (b.deconv.in_channels, b.deconv.out_channels) for b in expected.blocks]
            assert [b.deconv.stride for b in actual.blocks] == (
                [b.deconv.stride for b in expected.blocks] if target == "input"
                else [(1,)] * len(strides))
            out = model(torch.randn(2, 4, 64))
            assert out["recon"].shape == out["target"].shape
    # The configurable freeze flag governs parameters, gradients and BN mode.
    config["cross_attention_kwargs"]["freeze_encoder"] = False
    model = get_cross_attention_model(config).train()
    assert model.encoder.training
    assert all(param.requires_grad for param in model.encoder.parameters())
    model(torch.randn(2, 4, 64))["recon"].square().mean().backward()
    assert any(param.grad is not None and param.grad.abs().sum() > 0
               for param in model.encoder.parameters())
    print("Automatic mirror for all targets/depths/strides and freeze modes: OK")


def verify_builders():
    config = {"model_kwargs": {"in_channels": 4, "enc_channels": [8, 12],
                               "enc_strides": [2, 2], "dropout": 0.0},
              "aux_kwargs": {"with_aux": False},
              "predictor_kwargs": {"domain": "time", "time_steering": False,
                                   "stem_channels": 8, "neck_widths": [8, 8, 8]}}
    builders = (get_ae_model, get_vae_model, get_lewm_model,
                get_recon_model, get_cross_attention_model)
    x = torch.randn(2, 4, 64)
    for build in builders:
        assert build.__module__ == "models.builders"
        model = build(config).eval()
        assert model.encoder.output_dims == 12
        assert model.encoder.total_stride == 4
        if build in (get_ae_model, get_vae_model):
            assert model(x)["recon"].shape == x.shape
            assert model.variational == (build is get_vae_model)
        elif build is get_recon_model:
            assert model(x)["recon"].shape == x.shape
        elif build is get_cross_attention_model:
            out = model(x)
            assert out["recon"].shape == out["target"].shape
        else:
            assert model(x)["predicted"]["L0"].shape == (2, 12, 16)
    # Existing imports remain aliases, preserving notebooks and external callers.
    from lewm import get_lewm_model as old_lewm
    from lewm_reconstruction import get_recon_model as old_recon
    from lewm_cross_attention import get_cross_attention_model as old_cross

    assert old_lewm is get_lewm_model and old_recon is get_recon_model
    assert old_cross is get_cross_attention_model
    print("All five model builders live in models/; training import aliases preserved: OK")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    verify_mirror()
    verify_targets_and_freezing()
    verify_builders()


if __name__ == "__main__":
    main()
