"""Shared LEWM lineage: inherited configs and both phase-two handoffs."""

from pathlib import Path
import csv
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lewm_cross_attention import main as attention_main
from lewm_reconstruction import main as reconstruction_main
from utils.common_config import get_criterion
from utils.config import create_config, experiment_base_dir, load_experiment_config


def expect_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def verify_configs(root):
    for phase1 in (REPO / "configs/lewm_encoder").glob("*/phase1.yml"):
        parent = load_experiment_config(phase1)
        for child in sorted(phase1.parent.glob("*/*.yml")):
            config = load_experiment_config(child)
            assert config["model_kwargs"] == parent["model_kwargs"]
            assert config["aux_kwargs"] == parent["aux_kwargs"]
            assert config["phase1_experiment"] == parent["phase1_experiment"]
            assert config["tag_phase1"] == parent["tag_jepa"]
            # Inheriting phase-one training must not leave loss args behind.
            assert get_criterion(config) is not None
            if config["criterion"] == "recon_l1":
                assert config["criterion_kwargs"] == {}

    cycle = root / "cycle.yml"
    cycle.write_text("extends: cycle.yml\n")
    expect_error(lambda: load_experiment_config(cycle))
    base = {"train_db_name": "smd", "framework": "cross_attention",
            "phase1_experiment": "time", "experiment_name": "default"}
    path = experiment_base_dir(root, base, "machine-1-1.txt", "run1")
    assert Path(path) == root / "smd/lewm_encoder/time/cross_attention/default/run1/machine-1-1.txt"
    assert path != experiment_base_dir(root, {**base, "phase1_experiment": "frequency"},
                                       "machine-1-1.txt", "run1")
    expect_error(lambda: experiment_base_dir(root, {**base, "phase1_experiment": "../time"},
                                             "machine-1-1.txt", "run1"))


def verify_shared_training(root):
    env = root / "env.yml"
    env.write_text(yaml.safe_dump({"root_dir": str(root / "results")}))
    configs = REPO / "configs/lewm_encoder/time"
    base = {"train_db_name": "synthetic", "val_db_name": "synthetic",
            "synthetic_kwargs": {"n_steps": 640, "test_n_steps": 240, "n_channels": 4},
            "val_fraction": 0.25, "wsz": 64, "stride": 32, "batch_size": 4,
            "epochs": 1, "device": "cpu", "amp": False, "num_workers": 0,
            "eval_window_size": 8, "score_batch_size": 8,
            "model_kwargs": {"in_channels": 4, "enc_channels": [8, 12],
                             "enc_strides": [2, 2], "norm": "batch", "dropout": 0.0}}
    first = SimpleNamespace(config_env=str(env), config_exp=str(configs / "phase1.yml"),
                            fname="toy", version="shared")
    reconstruction_main(first, {**base, "predictor_kwargs": {"domain": "time", "time_steering": False,
                                                           "stem_channels": 8, "neck_widths": [8, 8, 8]}})
    first_config = create_config(str(env), first.config_exp, "toy", "shared", base)
    source = Path(first_config["jepa_model"])
    original = source.read_bytes()
    source_state = torch.load(source, weights_only=False)
    for runner, child, extra in (
        (reconstruction_main, "reconstruction/default.yml", {"stage_c": {"mode": "frozen", "eval_every": 0}}),
        (attention_main, "cross_attention/observed_input.yml", {
            "cross_attention_kwargs": {"target": "input", "query_source": "observed",
                                       "qk_channels": 4, "value_channels": 4,
                                       "context_crop": None, "target_crop": None,
                                       "dropout": 0.0}}),
    ):
        args = SimpleNamespace(config_env=str(env), config_exp=str(configs / child),
                               fname="toy", version="phase2_run", phase1_version="shared")
        patch = {**base, **extra, "phase1_version": "shared"}
        p = create_config(str(env), args.config_exp, "toy", args.version, patch)
        assert p["pretrained_from"] == str(source)  # no explicit weight path
        runner(args, patch)
        state = torch.load(p["jepa_checkpoint"], weights_only=False)
        assert state["next_epoch"] == 1
        assert all(torch.equal(value, source_state[key]) for key, value in state["model"].items()
                   if key.startswith("encoder."))
        assert source.read_bytes() == original  # neither child changes the source
        args.stage = "score"
        report = runner(args, patch)
        assert report["phase1_experiment"] == "time"
        assert report["experiment_name"] in ("default", "observed_input")
        assert Path(p["metrics_path"]).is_file()


def verify_aggregation(root):
    from aggregate_lewm_metrics import main as aggregate

    source = root / "aggregate_source"
    dest = root / "aggregated"
    groups = ("time/reconstruction/default/run1", "frequency/reconstruction/default/run1",
              "time/cross_attention/observed_input/run1")
    for group in groups:
        for machine, value in (("machine-1-1.txt", 0.5), ("machine-1-2.txt", 0.7)):
            directory = source / group / machine / "jepa"
            directory.mkdir(parents=True)
            (directory / "metrics.json").write_text(json.dumps({"honest": {"f1": value}}))
    with patch.object(sys, "argv", ["aggregate", "--base", str(source), "--outdir", str(dest)]):
        assert aggregate() == 0
    with (dest / "summary_mean.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["experiment"] for row in rows} == set(groups)
    assert all(abs(float(row["honest/f1"]) - 0.6) < 1e-6 for row in rows)
    assert all((dest / (group + ".csv")).is_file() for group in groups)


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="lewm-layout-") as tmp:
        root = Path(tmp)
        verify_configs(root)
        verify_shared_training(root)
        verify_aggregation(root)
    print("Inherited configs and automatic shared phase-one handoff to both frameworks: OK")


if __name__ == "__main__":
    main()
