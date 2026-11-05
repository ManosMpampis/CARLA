"""Plain assertions for causal query extraction and the existing runner contract."""

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.builders import get_cross_attention_model, get_lewm_model
from models.lewm_cross_attention_tcn import CausalQueryTCN, CrossAttentionTCNLeWM
from lewm_cross_attention import main as run_stage
from utils.common_config import get_criterion
from utils.config import experiment_base_dir, load_experiment_config


def expect_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def configuration(target="features"):
    return {
        "model_kwargs": {"in_channels": 4, "enc_channels": [8, 12],
                         "enc_strides": [2, 2], "norm": "batch", "dropout": 0.0},
        "cross_attention_model": "lewm_cross_attention_tcn",
        "cross_attention_kwargs": {"target": target, "query_source": "observed",
                                   "feature_extraction": "separate", "qk_channels": 4,
                                   "value_channels": 6, "num_heads": 2,
                                   "query_tcn_hidden_channels": 8,
                                   "query_tcn_dropout": 0.0, "dropout": 0.0},
        "criterion": "lewm_cross_attention",
        "criterion_kwargs": {"sigreg_kwargs": {"num_slices": 4}},
    }


def verify_causal_features():
    tcn = CausalQueryTCN(4, 6, hidden_channels=8, dropout=0.0)
    assert tcn.receptive_field == 31
    x = torch.randn(2, 4, 64, requires_grad=True)
    for training in (True, False):
        tcn.train(training)
        original = tcn(x)
        changed = x.detach().clone()
        changed[..., 33:] += torch.randn_like(changed[..., 33:]) * 10
        assert original.shape == (2, 6, 64)
        torch.testing.assert_close(original[..., :33], tcn(changed)[..., :33], rtol=0, atol=0)
    gradient = torch.autograd.grad(tcn(x)[..., 40].sum(), x)[0]
    assert gradient[..., 41:].count_nonzero() == 0
    assert gradient[..., :10].count_nonzero() == 0
    assert gradient[..., 10:41].abs().sum() > 0
    for patch in ({"dilations": []}, {"dilations": [0]}, {"kernel_size": 0},
                  {"hidden_channels": 0}, {"dropout": 1.0}):
        expect_error(lambda: CausalQueryTCN(4, 6, **patch))
    print("Feature-token causality and finite receptive field: OK")


def verify_model():
    for target in ("features", "input", "query"):
        p = configuration(target)
        model = get_cross_attention_model(p).train()
        assert isinstance(model, CrossAttentionTCNLeWM)
        assert not model.encoder.training
        x = torch.randn(3, 4, 64)
        out = model(x, mask={"X_inj": x + torch.randn_like(x) * 0.2})
        assert out["recon"].shape == out["target"].shape
        loss = get_criterion(p)(out)["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.qkv_encoder.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert sum(p.grad.abs().sum() for p in model.qkv_encoder.query.parameters()) > 0
        assert all(p.grad is None for p in model.encoder.parameters())
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        model.eval()
        before = model(x)
        changed = x.clone()
        changed[..., 32:] += torch.randn_like(changed[..., 32:]) * 4
        after = model(changed)
        assert torch.equal(before["qkv"]["key"], after["qkv"]["key"])
        assert torch.equal(before["qkv"]["value"], after["qkv"]["value"])
        assert not torch.equal(before["qkv"]["query"], after["qkv"]["query"])
        assert model.score(x)["fused"].shape == (3, 32)
        saved = copy.deepcopy(model.state_dict())
        restored = get_cross_attention_model(p).eval()
        restored.load_state_dict(saved)
        torch.testing.assert_close(before["recon"], restored(x)["recon"])
        altered = copy.deepcopy(p)
        altered["cross_attention_kwargs"]["query_tcn_dilations"] = [1, 2, 4, 4]
        expect_error(lambda: get_cross_attention_model(altered).load_state_dict(saved))
        baseline = copy.deepcopy(p)
        baseline.pop("cross_attention_model")
        for key in ("query_tcn_hidden_channels", "query_tcn_dropout"):
            baseline["cross_attention_kwargs"].pop(key)
        expect_error(lambda: get_cross_attention_model(baseline).set_extra_state(model.get_extra_state()))
    invalid = configuration()
    invalid["cross_attention_kwargs"]["query_source"] = "context"
    expect_error(lambda: get_cross_attention_model(invalid))
    paths = sorted((REPO / "configs/lewm_encoder").glob("*/cross_attention/observed_features_tcn.yml"))
    assert len(paths) == 6
    for path in paths:
        p = load_experiment_config(path)
        p["model_kwargs"] = configuration()["model_kwargs"]
        assert isinstance(get_cross_attention_model(p), CrossAttentionTCNLeWM)
        assert p["cross_attention_kwargs"]["feature_extraction"] == "separate"
    print("Registry/config construction, gradients, crops, scoring and checkpoint semantics: OK")


def verify_runner():
    with tempfile.TemporaryDirectory(prefix="lewm-tcn-") as tmp:
        directory = Path(tmp)
        env = directory / "env.yml"
        env.write_text(yaml.safe_dump({"root_dir": str(directory / "results")}))
        source = directory / "encoder.pth.tar"
        torch.save(get_lewm_model(configuration()).state_dict(), source)
        path = REPO / "configs/lewm_encoder/time/cross_attention/observed_features_tcn.yml"
        patch = {**configuration(), "train_db_name": "synthetic", "val_db_name": "synthetic",
                 "synthetic_kwargs": {"n_steps": 256, "test_n_steps": 160, "n_channels": 4},
                 "val_fraction": 0.3, "wsz": 64, "stride": 16, "batch_size": 2,
                 "epochs": 1, "device": "cpu", "num_workers": 0, "amp": False,
                 "eval_every": 0, "pretrained_from": str(source),
                 "stage_b": {"masking": {"mode": "none"}}, "score_batch_size": 8,
                 "eval_window_size": 8}
        args = SimpleNamespace(config_env=str(env), config_exp=str(path), fname="toy", version="smoke")
        run_stage(args, patch)
        p = {**load_experiment_config(path), **patch}
        output = Path(experiment_base_dir(directory / "results", p, "toy", "smoke")) / "jepa_cross_attention_observed_features_tcn"
        assert (output / "model.pth.tar").is_file()
        state = torch.load(output / "checkpoint.pth.tar", weights_only=False)
        assert state["next_epoch"] == 1
        run_stage(args, patch)  # Existing checkpoint resumes without another training epoch.
        run_stage(args, {**patch, "stage": "score"})
        assert (output / "calibration.json").is_file()
        assert (output / "metrics.json").is_file()
    print("Synthetic train -> checkpoint resume -> scoring export: OK")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(4)
    verify_causal_features()
    verify_model()
    verify_runner()
    print("verify_lewm_cross_attention_tcn: OK")
