"""Single reconstruction entry: pretrain -> recon -> resume -> score."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import experiments_psm
import experiments_rec
from lewm import run_pretrain
from lewm_reconstruction import STAGES, main as run_stage, run_recon
from models.builders import get_recon_model
from utils.config import create_config, load_experiment_config, model_path


def main():
    torch.set_num_threads(1)
    assert all(STAGES[key] is run_pretrain for key in ("pretrain", "pretext", "phase1"))
    assert all(STAGES[key] is run_recon for key in ("recon", "phase2", "adapt"))
    # Both sweep scripts send all three stages to the same framework entry.
    for module in (experiments_psm, experiments_rec):
        with patch.object(module, "recon_main") as runner:
            if module is experiments_psm:
                module.run("phase1", "phase2")
            else:
                module.run_machine("machine-1-1.txt", "phase1", "phase2")
            assert [call.kwargs["update_dictionary"]["stage"]
                    for call in runner.call_args_list] == ["pretrain", "recon", "score"]
            assert not getattr(runner.call_args_list[1].args[0], "score", False)
            assert runner.call_args_list[2].args[0].score
            assert runner.call_args_list[1].args[0].config_exp == runner.call_args_list[2].args[0].config_exp
            auxiliary = [call.kwargs["update_dictionary"]["aux_kwargs"]
                         for call in runner.call_args_list]
            assert auxiliary[0] == auxiliary[1] == auxiliary[2]

    with tempfile.TemporaryDirectory(prefix="lewm-reconstruction-") as tmp:
        root = Path(tmp)
        env = root / "env.yml"
        env.write_text(yaml.safe_dump({"root_dir": str(root / "results")}))
        base = {"train_db_name": "synthetic", "val_db_name": "synthetic",
                "synthetic_kwargs": {"n_steps": 640, "test_n_steps": 240, "n_channels": 4},
                "val_fraction": 0.25, "wsz": 64, "stride": 32, "batch_size": 4,
                "epochs": 1, "device": "cpu", "amp": False, "num_workers": 0,
                "eval_window_size": 8, "score_batch_size": 8,
                "model_kwargs": {"in_channels": 4, "enc_channels": [8, 12],
                                 "enc_strides": [2, 2], "norm": "batch", "dropout": 0.0}}
        first = SimpleNamespace(config_env=str(env),
                                config_exp=str(REPO / "configs/lewm_encoder/time/phase1.yml"),
                                fname="toy", version="phase1", stage="phase1")
        run_stage(first, {**base, "predictor_kwargs": {"domain": "time", "time_steering": False,
                                                     "stem_channels": 8, "neck_widths": [8, 8, 8]}})
        source = Path(model_path(root / "results", {**load_experiment_config(first.config_exp), **base}, "toy", "phase1"))
        assert source.is_file()
        first.stage = "score"
        first.score_checkpoint = str(source)
        phase1_report = run_stage(first, {**base,
            "predictor_kwargs": {"domain": "time", "time_steering": False,
                                 "stem_channels": 8, "neck_widths": [8, 8, 8]},
            "probe_kwargs": {"num_probe_windows": 0}})
        assert Path(phase1_report["decision_trace"]["html"]).is_file()
        second = SimpleNamespace(config_env=str(env),
                                 config_exp=str(REPO / "configs/lewm_encoder/frequency_aux/reconstruction/default.yml"),
                                 fname="toy", version="phase2", stage="phase2",
                                 pretrained_from=str(source))
        overrides = {**base, "phase1_experiment": "time", "tag_phase1": "time_predictor", "stage_c": {"mode": "frozen", "eval_every": 0},
                     "recon_kwargs": {"with_aux": False, "dropout": 0.0}}
        run_stage(second, overrides)
        p = create_config(str(env), second.config_exp, "toy", "phase2", overrides)
        checkpoint = torch.load(p["jepa_checkpoint"], weights_only=False)
        phase1_state = torch.load(source, weights_only=False)
        assert checkpoint["next_epoch"] == 1
        assert all(torch.equal(value, phase1_state[key])
                   for key, value in checkpoint["model"].items() if key.startswith("encoder."))
        model = get_recon_model(p)
        head_before = {key: value.clone() for key, value in model.head.state_dict().items()}
        model.load_state_dict(checkpoint["model"], strict=True)
        assert any(not torch.equal(value, head_before[key]) for key, value in model.head.state_dict().items())

        # Resume has all encoder/head weights and no dependency on the old file.
        second.pretrained_from = str(root / "missing-phase1")
        run_stage(second, {**overrides, "epochs": 2})
        checkpoint = torch.load(p["jepa_checkpoint"], weights_only=False)
        assert checkpoint["next_epoch"] == 2
        second.stage = "score"
        second.score_checkpoint = p["jepa_model"]
        report = run_stage(second, overrides)
        assert set(report["evaluation"]) == {"window", "timeseries"}
        assert Path(report["decision_trace"]["html"]).is_file()
        assert Path(report["decision_trace"]["json"]).is_file()
        assert Path(p["scores_path"]).is_file()
        calibration = json.loads(Path(p["calibration_path"]).read_text())
        assert calibration["source"] == "held-out validation tail"
    print("Single LEWM reconstruction entry: both phases, frozen transfer, resume, scoring, sweeps: OK")


if __name__ == "__main__":
    main()
