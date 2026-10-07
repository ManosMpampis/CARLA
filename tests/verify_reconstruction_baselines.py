"""Plain-Python smoke and score-contract checks for the AE/VAE arms."""

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from models.ae_baseline import ReconstructionBaseline, reconstruction_objective
from carla_recon import get_recon_model, run_score
from utils.config import create_config
from utils.reconstruction_scores import reconstruction_score_map
from utils.reconstruction_baselines import score_both, train_arm


def main():
    residual = torch.tensor([[[3.0, 0.0], [4.0, 2.0]]])
    zero = torch.zeros_like(residual)
    for mode, expected in (
        ("l1", [3.5, 1.0]),
        ("l2", [5.0 / np.sqrt(2.0), np.sqrt(2.0)]),
        ("mse", [12.5, 2.0]),
    ):
        scores, _ = reconstruction_score_map(residual, zero, mode)
        np.testing.assert_allclose(scores.numpy()[0], expected, rtol=1e-6)

    kwargs = {"in_channels": 2, "enc_channels": [4, 8],
              "enc_strides": [2, 2], "norm": "batch", "dropout": 0.0}
    x = torch.randn(2, 2, 16)
    for variational in (False, True):
        model = ReconstructionBaseline(kwargs, {"dropout": 0.0}, variational)
        model.eval()
        out = model(x, sample=False)
        assert out["recon"].shape == x.shape
        loss = reconstruction_objective(out, x, 0.3)
        assert torch.isfinite(loss["loss"])
        assert (loss["kl"] > 0) == variational
        loss["loss"].backward()
        assert model.encoder.blocks[0].main[0].conv.weight.grad is not None
        if variational:
            assert model.mu.weight.grad is not None
            assert model.logvar.weight.grad is not None
        for mode in ("l1", "l2", "mse"):
            model.score_mode = mode
            score = model.score(x)["fused"]
            expected, _ = reconstruction_score_map(out["recon"], x, mode)
            torch.testing.assert_close(score, expected)

    for mode in ("l1", "l2", "mse"):
        model = get_recon_model({"model_kwargs": kwargs,
                                 "recon_kwargs": {"dropout": 0.0},
                                 "score_mode": mode})
        model.eval()
        expected, _ = reconstruction_score_map(model.reconstruct(x), x, mode)
        torch.testing.assert_close(model.score(x)["fused"], expected)

    class ZeroModel:
        def eval(self):
            return self

        def score(self, batch):
            # Two channels: mean channel squared error, one score per point.
            return {"fused": batch.square().mean(dim=1)}

    series = np.arange(12, dtype=np.float32).reshape(6, 2)
    scored = score_both(ZeroModel(), series, window=4, stride=2,
                        batch_size=2, device="cpu")
    expected = np.square(series).mean(axis=1)
    np.testing.assert_allclose(scored["timeseries_scores"], expected)
    np.testing.assert_allclose(scored["window_scores"],
                               [expected[:4].mean(), expected[2:].mean()])
    np.testing.assert_array_equal(scored["cover_counts"], [1, 1, 2, 2, 1, 1])

    with tempfile.TemporaryDirectory() as root:
        env = os.path.join(root, "env.yml")
        with open(env, "w") as stream:
            yaml.safe_dump({"root_dir": root}, stream)
        for arm in ("ae", "vae"):
            with open(f"configs/baselines/smd_{arm}.yml") as stream:
                config = yaml.safe_load(stream)
            config.update({
                "train_db_name": "synthetic", "val_db_name": "synthetic",
                "synthetic_kwargs": {"n_steps": 320, "test_n_steps": 240,
                                     "n_channels": 2},
                "model_kwargs": kwargs, "recon_kwargs": {"dropout": 0.0},
                "wsz": 16, "stride": 4, "val_fraction": 0.25,
                "device": "cpu", "epochs": 1, "batch_size": 8,
                "score_batch_size": 32, "eval_window_size": 8,
            })
            path = os.path.join(root, f"{arm}.yml")
            with open(path, "w") as stream:
                yaml.safe_dump(config, stream)
            train_arm(arm, SimpleNamespace(config_env=env, config_exp=path,
                                           fname="toy", version=arm))
            run_dir = os.path.join(root, "synthetic", arm, "toy", f"jepa_{arm}")
            files = ["best_validation_loss.pth.tar"]
            for protocol in ("window", "timeseries"):
                files.extend(f"best_test_{protocol}_{key}.pth.tar" for key in
                             ("calibrated_f1", "oracle_f1", "vus_pr", "vus_roc"))
            for filename in files:
                checkpoint = torch.load(os.path.join(run_dir, filename),
                                        map_location="cpu", weights_only=False)
                assert checkpoint["epoch"] == 1
                assert checkpoint["inference"]["window_length"] == 16
                assert checkpoint["inference"]["stride"] == 4
                assert "train" in checkpoint["losses"]
                assert "window" in checkpoint["evaluation"]
                assert "timeseries" in checkpoint["evaluation"]
            assert os.path.exists(os.path.join(run_dir, "tensorboard"))
        steering_weights = os.path.join(root, "steering_weights.pth.tar")
        torch.save(get_recon_model({"model_kwargs": kwargs,
                                   "recon_kwargs": {"dropout": 0.0}}).state_dict(),
                   steering_weights)
        config.update({"stage": "score", "tag_jepa": "steering_smoke",
                       "score_checkpoint": steering_weights,
                       "score_mode": "mse",
                       "calibration_kwargs": {"quantile": 0.99}})
        steering_config = os.path.join(root, "steering.yml")
        with open(steering_config, "w") as stream:
            yaml.safe_dump(config, stream)
        steering_run = create_config(env, steering_config, "toy", "steering")
        report = run_score(steering_run, torch.device("cpu"))
        assert report["score_mode"] == "mse"
        for protocol in ("window", "timeseries"):
            assert "calibrated" in report["evaluation"][protocol]
            assert "oracle" in report["evaluation"][protocol]
            assert "vus_pr" in report["evaluation"][protocol]
        saved = np.load(steering_run["scores_path"])
        assert len(saved["window_scores"]) == len(saved["window_predictions"])
        assert len(saved["timeseries_scores"]) == len(saved["timeseries_predictions"])
    print("AE/VAE reconstruction baseline verification passed")


if __name__ == "__main__":
    main()
