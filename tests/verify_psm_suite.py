"""PSM suite assertions using a temporary PSM-format dataset and all runners."""

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.jepa_dataset import JEPADataset, make_synthetic_series
from utils.config import create_config, experiment_base_dir
from utils.experiment_suite import (build_plan, checkpoint_path, experiment_dir,
                                    run_suite)


def expect_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def verify_plan(directory):
    env = directory / "env.yml"
    env.write_text(yaml.safe_dump({"root_dir": str(directory / "results")}))
    original_manifest = REPO / "configs/psm/experiments.yml"
    plan, root = build_plan(original_manifest, env, "trial")
    assert len(plan) == 16 and len({exp.key for exp in plan}) == 16
    assert {exp.framework for exp in plan} == {
        "ae", "vae", "lewm_encoder", "reconstruction", "cross_attention"}
    assert all(exp.config["model_kwargs"]["in_channels"] == 25 for exp in plan)
    assert not Path(root).exists()  # planning/dry-run is read-only
    selected, _ = build_plan(original_manifest, env, "trial",
                              experiments=["lewm_encoder/time/cross_attention/context_input"])
    assert [exp.key for exp in selected] == ["lewm_encoder/time", "lewm_encoder/time/cross_attention/context_input"]
    assert selected[1].config["pretrained_from"] == str(checkpoint_path(selected[0], root, "trial"))
    selected, _ = build_plan(original_manifest, env, "trial", frameworks=["vae"], epochs=3)
    assert [exp.key for exp in selected] == ["vae/default"]
    assert selected[0].config["epochs"] == 3
    expect_error(lambda: build_plan(original_manifest, env, "trial", experiments=["vae/missing"]))
    expect_error(lambda: build_plan(original_manifest, env, "../trial"))
    expect_error(lambda: build_plan(original_manifest, env, "trial", epochs=0))

    variant_manifest = yaml.safe_load(original_manifest.read_text())
    for entry in variant_manifest["experiments"]:
        entry["config"] = str((original_manifest.parent / entry["config"]).resolve())
    variant_manifest["experiments"].extend([
        {"framework": "lewm_encoder", "experiment_name": "time_lr_0.1", "runner": "lewm",
         "config": str(REPO / "configs/lewm_encoder/time/phase1.yml"),
         "overrides": {"optimizer_kwargs": {"lr": 0.1}}},
        {"framework": "reconstruction", "experiment_name": "default", "runner": "recon",
         "config": str(REPO / "configs/lewm_encoder/time/reconstruction/default.yml"),
         "pretrained_experiment": "lewm_encoder/time_lr_0.1"},
    ])
    variant_path = directory / "named_encoder.yml"
    variant_path.write_text(yaml.safe_dump(variant_manifest))
    variant_plan, _ = build_plan(variant_path, env, "trial",
                                experiments=["lewm_encoder/time_lr_0.1/reconstruction/default"])
    assert [exp.key for exp in variant_plan] == ["lewm_encoder/time_lr_0.1",
                                                "lewm_encoder/time_lr_0.1/reconstruction/default"]
    assert all(exp.config["phase1_experiment"] == "time_lr_0.1" for exp in variant_plan)
    assert variant_plan[1].config["pretrained_from"] == str(checkpoint_path(variant_plan[0], root, "trial"))

    cfg = {"train_db_name": "psm", "framework": "vae", "experiment_name": "lr_0.1"}
    expected = Path(root) / "psm/vae/lr_0.1/trial/psm"
    assert Path(experiment_base_dir(root, cfg, "psm", "trial")) == expected
    assert Path(experiment_base_dir(root, {"train_db_name": "psm"}, "psm", "legacy/run")) == (
        Path(root) / "psm/legacy/run/psm")
    expect_error(lambda: experiment_base_dir(root, {**cfg, "experiment_name": "../bad"}, "psm", "trial"))
    print("16-arm plan, dependency ordering, named/legacy paths and validation: OK")
    return env


def fixture_manifest(directory):
    manifest_path = REPO / "configs/psm/experiments.yml"
    manifest = yaml.safe_load(manifest_path.read_text())
    for entry in manifest["experiments"]:
        entry["config"] = str((manifest_path.parent / entry["config"]).resolve())
    common = manifest["common"]
    common.update(epochs=1, device="cpu", wsz=128, stride=32, val_fraction=0.25,
                  batch_size=4, score_batch_size=8, eval_window_size=8,
                  dataset_root=str(directory / "PSM"))
    common["model_kwargs"].update(enc_channels=[4, 8], dropout=0.0)
    common["predictor_kwargs"] = {"stem_channels": 4, "neck_widths": [4, 4, 4],
                                   "n_fft": 16, "hop_length": 4, "win_length": 16}
    common["aux_kwargs"] = {"aux_channels": [4, 4, 4]}
    common["cross_attention_kwargs"] = {"qk_channels": 4, "value_channels": 4,
                                        "dropout": 0.0}
    common["criterion_kwargs"] = {"sigreg_kwargs": {"num_slices": 4}}
    common["recon_kwargs"] = {"dropout": 0.0}
    # recon_l1 has no SIGReg argument; remove the common criterion override
    # and attach the tiny regularizer only to LEWM-based entries.
    criterion = common.pop("criterion_kwargs")
    for entry in manifest["experiments"]:
        if entry["runner"] in ("lewm", "cross_attention"):
            entry.setdefault("overrides", {})["criterion_kwargs"] = criterion
    manifest["experiments"].append({"framework": "vae", "experiment_name": "lr_0.1",
                                    "runner": "vae", "config": str(REPO / "configs/baselines/smd_vae.yml"),
                                    "overrides": {"optimizer_kwargs": {"lr": 0.1}}})
    # Same phase-two names beneath another encoder must have separate paths/rows.
    manifest["experiments"].extend([
        {"framework": "reconstruction", "experiment_name": "default", "runner": "recon",
         "config": str(REPO / "configs/lewm_encoder/time/reconstruction/default.yml"),
         "pretrained_experiment": "lewm_encoder/time"},
        {"framework": "cross_attention", "experiment_name": "observed_input", "runner": "cross_attention",
         "config": str(REPO / "configs/lewm_encoder/frequency_aux/cross_attention/observed_input.yml"),
         "pretrained_experiment": "lewm_encoder/frequency_aux", "overrides": {"criterion_kwargs": criterion}},
    ])
    path = directory / "experiments.yml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return path, manifest


def make_psm(directory):
    root = directory / "PSM"
    root.mkdir()
    columns = [f"feature_{i}" for i in range(25)]
    train, _ = make_synthetic_series(1024, 25, 4)
    test, _ = make_synthetic_series(512, 25, 5)
    labels = np.zeros(512, dtype=np.int64)
    for a, b in ((192, 224), (400, 420)):
        test[a:b, :3] += 6
        labels[a:b] = 1
    for name, values in (("train", train), ("test", test)):
        frame = pd.DataFrame(values, columns=columns)
        frame.insert(0, "timestamp_(min)", np.arange(len(frame)))
        frame.to_csv(root / f"{name}.csv", index=False)
    pd.DataFrame({"timestamp_(min)": np.arange(512), "label": labels}).to_csv(
        root / "test_label.csv", index=False)


def verify_runs(directory, env):
    make_psm(directory)
    manifest_path, manifest = fixture_manifest(directory)
    plan, root = build_plan(manifest_path, env, "run1")
    assert len(plan) == 19
    assert len({exp.key for exp in plan}) == 19
    assert len({experiment_dir(exp, root, "run1") for exp in plan}) == 19
    lr_variant = next(exp for exp in plan if exp.key == "vae/lr_0.1")
    assert lr_variant.config["optimizer_kwargs"] == {"lr": 0.1, "weight_decay": 0.01}
    p = create_config(env, lr_variant.config_path, "psm", "run1",
                      update_dictionary=lr_variant.config)
    assert Path(p["jepa_dir"]) == experiment_dir(lr_variant, root, "run1")
    data = JEPADataset(p, train=True)
    test_data = JEPADataset(p, train=False)
    assert data.series.shape == (768, 25)
    assert data.val_series.shape == (256, 25)
    assert test_data.series.shape == (512, 25)
    np.testing.assert_array_equal(test_data.targets[192:224], np.ones(32))
    train_frame = pd.read_csv(directory / "PSM/train.csv").iloc[:, 1:].to_numpy()
    np.testing.assert_allclose(data.mean, train_frame.mean(axis=0), rtol=1e-6, atol=1e-6)
    rows = run_suite(plan, root, env, "run1")
    assert len(rows) == 19
    assert all(row["status"] == "completed" for row in rows), rows
    for exp in plan:
        out = experiment_dir(exp, root, "run1")
        report = json.loads((out / "metrics.json").read_text())
        assert report["selection"]["source"] == "validation loss"
        assert report["experiment_key"] == exp.key
        assert Path(report["selection"]["weights"]).name in (
            "model.pth.tar", "best_validation_loss.pth.tar")
        assert {"honest", "point_adjust_comparability", "evaluation"} <= set(report)
        assert Path(report["decision_trace"]["html"]).is_file()
        assert Path(report["decision_trace"]["json"]).is_file()
        assert (out / "resolved_config.yml").is_file()
        assert any(out.rglob("events.out.tfevents.*"))
        saved = np.load(out / "scores.npz")
        assert len(saved["timeseries_scores"]) == 512
        calibration = json.loads((out / "calibration.json").read_text())
        assert calibration["source"] == "held-out clean train tail"
        expected = ((saved["timeseries_scores"] > calibration["timeseries_threshold"])
                    & (saved["cover_counts"] > 0))
        np.testing.assert_array_equal(saved["timeseries_predictions"], expected)
        if exp.runner == "cross_attention":
            assert report["coverage"]["output_window"] == 64
            assert (saved["cover_counts"][:64] == 0).all()
    summary = Path(root) / "psm/summaries/run1/summary.csv"
    with summary.open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 19
    assert len({row["experiment_key"] for row in table}) == 19
    assert {row["experiment_name"] for row in table if row["framework"] == "vae"} == {"default", "lr_0.1"}
    # Resume a subset; the summary must retain all other experiments.
    subset, _ = build_plan(manifest_path, env, "run1", experiments=["vae/default"], epochs=2)
    rows = run_suite(subset, root, env, "run1")
    assert rows[0]["status"] == "completed"
    checkpoint = torch.load(experiment_dir(subset[0], root, "run1") / "last.pth.tar", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert len(json.loads(summary.with_suffix(".json").read_text())) == 19
    run_suite(subset, root, env, "run1", score_only=True)
    # Same directory with a changed learning rate must fail before training.
    changed = copy.deepcopy(subset)
    changed[0].config["optimizer_kwargs"]["lr"] = 0.03
    failed = run_suite(changed, root, env, "run1")
    assert failed[0]["status"] == "failed" and "configuration changed" in failed[0]["error"]
    # A missing phase-one source blocks its dependants while unrelated arms run.
    selected, _ = build_plan(manifest_path, env, "missing",
                              experiments=["lewm_encoder/time/cross_attention/context_input", "ae/default"])
    failed = run_suite(selected, root, env, "missing", score_only=True)
    assert {row["experiment_key"]: row["status"] for row in failed} == {
        "lewm_encoder/time": "failed", "lewm_encoder/time/cross_attention/context_input": "blocked", "ae/default": "failed"}
    print("All five frameworks / 19 PSM-format experiments, distinct lineages, results, resume and failures: OK")


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    with tempfile.TemporaryDirectory(prefix="psm-suite-") as temp:
        directory = Path(temp)
        env = verify_plan(directory)
        verify_runs(directory, env)
    print("verify_psm_suite: OK")


if __name__ == "__main__":
    main()
