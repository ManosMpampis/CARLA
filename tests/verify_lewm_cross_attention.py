"""Plain assertions: gradients, crop placement, leakage, transfer, and full runs."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.builders import get_cross_attention_model, get_lewm_model
from lewm_cross_attention import load_feature_extractor, main as run_stage, scoring_options
from models.lewm_cross_attention import CrossAttentionPredictor
from utils.common_config import get_criterion
from utils.config import experiment_base_dir, load_experiment_config, model_path
from utils.reconstruction_baselines import score_both


def expect_error(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def config(target="features", source="observed", strides=(2, 2)):
    return {"model_kwargs": {"in_channels": 4, "enc_channels": [8, 12],
                             "enc_strides": list(strides), "norm": "batch", "dropout": 0.1},
            "cross_attention_kwargs": {"target": target, "query_source": source,
                                       "context_crop": [1, 9], "target_crop": [10, 16],
                                       "feature_extraction": "separate",
                                       "qk_channels": 4, "value_channels": 6,
                                       "kernel_size": 3, "num_heads": 2,
                                       "norm": "batch", "dropout": 0.1},
            "criterion": "lewm_cross_attention", "criterion_kwargs": {
                "loss2_kind": "l1", "lambda_sigreg": 0.1, "lambda_sigreg_tgt": 0.1,
                "sigreg_kwargs": {"num_slices": 4}}, "wsz": 64, "stride": 8}


def verify_models():
    for strides in ((1, 1), (2, 2)):
        for source in ("observed", "context"):
            for target in ("features", "input", "query"):
                p = config(target, source, strides)
                model = get_cross_attention_model(p).train()
                assert not model.encoder.training
                frozen = {k: v.clone() for k, v in model.encoder.state_dict().items()}
                x = torch.randn(3, 4, 64)
                mask = {"X_inj": x + torch.randn_like(x) * 0.2}
                out = model(x, mask=mask)
                s = model.total_stride
                channels = {"features": 12, "input": 4, "query": 4}[target]
                length = 6 * s if target == "input" else 6
                assert out["recon"].shape == out["target"].shape == (3, channels, length)
                assert out["qkv"]["key"].shape == (3, 4, 8)
                assert out["qkv"]["value"].shape == (3, 6, 8)
                assert out["qkv"]["query"].shape == (3, 4, 6)
                if target == "input":
                    assert torch.equal(out["target"], x[..., 10*s:16*s])
                if target == "query":
                    assert out["target"].requires_grad
                    out["target"].retain_grad()
                losses = get_criterion(p)(out)
                assert all(torch.isfinite(v) for v in losses.values())
                losses["loss"].backward()
                for name, param in model.qkv_encoder.named_parameters():
                    assert param.grad is not None and torch.isfinite(param.grad).all(), name
                    assert param.grad.abs().sum() > 0, name
                assert all(param.grad is not None for param in model.predictor.parameters())
                assert all(param.grad is None for param in model.encoder.parameters())
                if target == "query":
                    assert out["target"].grad.abs().sum() > 0
                torch.optim.AdamW(model.parameters(), lr=1e-3).step()
                assert all(torch.equal(value, frozen[key])
                           for key, value in model.encoder.state_dict().items())
                model.eval()
                base = model(x)
                changed = x.clone()
                changed[..., 10*s:16*s] += 10
                other = model(changed)
                # Isolated context never observes the second crop.
                assert torch.equal(base["qkv"]["key"], other["qkv"]["key"])
                assert torch.equal(base["qkv"]["value"], other["qkv"]["value"])
                if source == "context":
                    assert torch.equal(base["recon"], other["recon"])
                    predicted = model.predict_next(x[..., s:9*s])
                    assert torch.allclose(base["recon"], predicted, atol=1e-6)
                else:
                    assert not torch.equal(base["qkv"]["query"], other["qkv"]["query"])
                    expect_error(lambda: model.predict_next(x[..., s:9*s]))
                assert model.score(x)["fused"].shape == (3, 6*s)
                for mode in ("l1", "l2", "mse"):
                    model.score_mode = mode
                    assert torch.isfinite(model.score(x)["fused"]).all()
                model.train()
                model.score(x)
                assert model.training and not model.encoder.training

    # The attention computation has Q(second), K/V(first), with no Query skip.
    qkv = {"query": torch.randn(2, 4, 5), "key": torch.randn(2, 4, 7),
           "value": torch.randn(2, 6, 7)}
    predictor = CrossAttentionPredictor(torch.nn.Identity(), 4, 6, 1).eval()
    expected = (torch.softmax(qkv["query"].transpose(1, 2) @ qkv["key"] / 2, dim=-1)
                @ qkv["value"].transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(predictor(qkv), expected, atol=1e-6)
    zero_values = {**qkv, "value": torch.zeros_like(qkv["value"])}
    assert torch.count_nonzero(predictor(zero_values)) == 0

    full_config = config()
    full_config["cross_attention_kwargs"]["feature_extraction"] = "full"
    full_model = get_cross_attention_model(full_config).eval()
    x = torch.randn(3, 4, 64)
    assert torch.equal(full_model(x)["target"], full_model.encoder(x)[..., 10:16])
    bad = copy.deepcopy(full_config)
    bad["cross_attention_kwargs"]["query_source"] = "context"
    expect_error(lambda: get_cross_attention_model(bad))
    for patch in ({"kernel_size": 2}, {"num_heads": 3},
                  {"context_crop": [0, 11]}, {"target_crop": [10, 30]},
                  {"context_crop": [0.5, 9]}, {"target": "invalid"}):
        bad = config()
        bad["cross_attention_kwargs"].update(patch)
        expect_error(lambda: get_cross_attention_model(bad).crop_ranges(64))
    expect_error(lambda: full_model.crop_ranges(63))
    p = config()
    model = get_cross_attention_model(p)
    options = scoring_options(p, model)
    assert options["check_mask"] == [40, 64]
    expect_error(lambda: scoring_options({**p, "evaluation": {"check_mask": [0, 24]}}, model))
    expect_error(lambda: scoring_options({**p, "evaluation": {"latency_offset": 1}}, model))
    series = np.random.default_rng(4).normal(size=(150, 4)).astype(np.float32)
    options.pop("wsz")
    options.pop("stride")
    result = score_both(model, series, 64, 8, 3, torch.device("cpu"), **options)
    assert result["starts"][0] == 40 and result["ends"][0] == 64
    assert (result["cover_counts"][:40] == 0).all()
    assert result["ends"][-1] == 150
    assert np.array_equal(result["starts"] - result["input_starts"],
                          np.full(len(result["starts"]), 40))
    print("Model gradients, attention, leakage, and crop scoring: OK")


def verify_transfer(directory):
    phase1 = {"model_kwargs": config()["model_kwargs"],
              "aux_kwargs": {"with_aux": False},
              "predictor_kwargs": {"domain": "time", "time_steering": False,
                                   "stem_channels": 8, "neck_widths": [8, 8, 8]}}
    pretrain = get_lewm_model(phase1)
    source = directory / "phase1.pth.tar"
    torch.save({"model": pretrain.state_dict()}, source)
    model = get_cross_attention_model(config())
    load_feature_extractor(source, model)
    assert all(torch.equal(value, pretrain.encoder.state_dict()[key])
               for key, value in model.encoder.state_dict().items())
    bad = config()
    bad["model_kwargs"]["enc_channels"] = [9, 12]
    expect_error(lambda: load_feature_extractor(source, get_cross_attention_model(bad)), RuntimeError)
    state = model.state_dict()
    expect_error(lambda: get_cross_attention_model(config(source="context")).load_state_dict(state))
    incomplete = pretrain.state_dict()
    del incomplete[next(k for k in incomplete if k.startswith("encoder."))]
    torch.save(incomplete, source)
    expect_error(lambda: load_feature_extractor(source, model), RuntimeError)
    print("Strict phase-one transfer and task checkpoint guards: OK")


def verify_runs(directory):
    env = directory / "env.yml"
    env.write_text(yaml.safe_dump({"root_dir": str(directory / "results")}))
    base = {"train_db_name": "synthetic", "val_db_name": "synthetic",
            "synthetic_kwargs": {"n_steps": 640, "test_n_steps": 240, "n_channels": 4},
            "val_fraction": 0.25, "wsz": 64, "stride": 32, "batch_size": 4,
            "epochs": 1, "device": "cpu", "amp": False, "num_workers": 0,
            "eval_window_size": 8, "score_batch_size": 8}
    path1 = REPO / "configs/lewm_encoder/time/phase1.yml"
    phase1_model_kwargs = config()["model_kwargs"]
    run_stage(SimpleNamespace(config_env=str(env), config_exp=str(path1), fname="toy", version="phase1"),
              {**base, "model_kwargs": phase1_model_kwargs,
               "predictor_kwargs": {"domain": "time", "time_steering": False,
                                    "stem_channels": 8, "neck_widths": [8, 8, 8]}})
    source = Path(model_path(directory / "results", {**load_experiment_config(path1), **base}, "toy", "phase1"))
    assert source.is_file()
    for query_source in ("observed", "context"):
        for target in ("features", "input", "query"):
            path = REPO / f"configs/lewm_encoder/time/cross_attention/{query_source}_{target}.yml"
            patch = {**base, **config(target, query_source), "stride": 32,
                     "pretrained_from": str(source)}
            args = SimpleNamespace(config_env=str(env), config_exp=str(path),
                                   fname="toy", version="phase2")
            run_stage(args, patch)
            out_dir = Path(experiment_base_dir(directory / "results", {**load_experiment_config(path), **patch}, "toy", "phase2")) / f"jepa_cross_attention_{query_source}_{target}"
            checkpoint = torch.load(out_dir / "checkpoint.pth.tar", weights_only=False)
            assert checkpoint["next_epoch"] == 1
            # Resume works without the original encoder checkpoint.
            run_stage(args, {**patch, "epochs": 2, "pretrained_from": "/missing/phase1"})
            checkpoint = torch.load(out_dir / "checkpoint.pth.tar", weights_only=False)
            assert checkpoint["next_epoch"] == 2
            report = run_stage(args, {**patch, "stage": "score"})
            assert report["task"]["target"] == target
            assert Path(report["decision_trace"]["html"]).is_file()
            assert Path(report["decision_trace"]["json"]).is_file()
            assert report["task"]["query_source"] == query_source
            scores = np.load(out_dir / "scores.npz")
            assert (scores["cover_counts"][:40] == 0).all()
            assert (scores["start_idxs"] - scores["input_start_idxs"] == 40).all()
            calibration = json.loads((out_dir / "calibration.json").read_text())
            assert calibration["source"] == "held-out clean train tail"
            assert set(report["evaluation"]) == {"window", "timeseries"}
            assert {"f1_score", "Event F1", "MCC"} <= set(report["honest"])
            assert "f1_score" in report["point_adjust_comparability"]
            expected_flags = ((scores["timeseries_scores"] > calibration["timeseries_threshold"])
                              & (scores["cover_counts"] > 0))
            assert np.array_equal(scores["timeseries_predictions"], expected_flags)
            assert (out_dir / "log.txt").exists()
            assert any(out_dir.rglob("events.out.tfevents.*"))
    print("Phase1 -> all six phase2 arms -> resume -> calibration/metrics/TensorBoard: OK")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    verify_models()
    with tempfile.TemporaryDirectory(prefix="lewm-cross-attention-") as temp:
        directory = Path(temp)
        verify_transfer(directory)
        verify_runs(directory)
    print("verify_lewm_cross_attention: OK")


if __name__ == "__main__":
    main()
