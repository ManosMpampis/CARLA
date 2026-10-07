"""Every entry scores its training config without running another epoch."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from carla_ae import main as ae_main
from carla_vae import main as vae_main
from lewm import main as encoder_main
from lewm_cross_attention import main as attention_main
from lewm_reconstruction import main as reconstruction_main
from utils.config import create_config, load_experiment_config


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="score-cli-") as tmp:
        root = Path(tmp)
        env = root / "env.yml"
        env.write_text(yaml.safe_dump({"root_dir": str(root / "results")}))
        base = {"train_db_name": "synthetic", "val_db_name": "synthetic",
                "synthetic_kwargs": {"n_steps": 640, "test_n_steps": 240, "n_channels": 4},
                "val_fraction": 0.25, "wsz": 64, "stride": 32, "batch_size": 4,
                "epochs": 1, "device": "cpu", "amp": False, "num_workers": 0,
                "eval_window_size": 8, "score_batch_size": 8, "save_timeseries_plot": False,
                "probe_kwargs": {"num_probe_windows": 0},
                "model_kwargs": {"in_channels": 4, "enc_channels": [8, 12],
                                 "enc_strides": [2, 2], "norm": "batch", "dropout": 0.0}}
        experiments = (
            (encoder_main, "lewm.py", "configs/lewm_encoder/time/phase1.yml", {
                "predictor_kwargs": {"domain": "time", "time_steering": False,
                                     "stem_channels": 8, "neck_widths": [8, 8, 8]}}),
            (reconstruction_main, "lewm_reconstruction.py", "configs/lewm_encoder/time/reconstruction/default.yml", {
                "stage_c": {"mode": "frozen", "eval_every": 0}}),
            (attention_main, "lewm_cross_attention.py", "configs/lewm_encoder/time/cross_attention/observed_input.yml", {
                "cross_attention_kwargs": {"target": "input", "query_source": "observed",
                                           "qk_channels": 4, "value_channels": 4,
                                           "context_crop": None, "target_crop": None,
                                           "dropout": 0.0}}),
            (ae_main, "carla_ae.py", "configs/baselines/smd_ae.yml", {}),
            (vae_main, "carla_vae.py", "configs/baselines/smd_vae.yml", {}),
        )
        process_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "MPLCONFIGDIR": str(root / "matplotlib")}
        configurations = {}
        for runner, entry, source, overrides in experiments:
            config = {**load_experiment_config(REPO / source), **base, **overrides}
            path = root / entry.replace(".py", ".yml")
            path.write_text(yaml.safe_dump(config))
            args = SimpleNamespace(config_env=str(env), config_exp=str(path), fname="toy", version="run1")
            runner(args)
            p = create_config(str(env), str(path), "toy", "run1")
            directory = Path(p["jepa_dir"])
            saved = {file: file.read_bytes() for file in directory.glob("*.pth.tar")}
            assert saved
            # If --score accidentally enters training, this extension changes
            # the checkpoint. Scoring must ignore remaining training epochs.
            config["epochs"] = 2
            path.write_text(yaml.safe_dump(config))
            command = [str(REPO / "venv/bin/python"), str(REPO / entry), "--config_env", str(env),
                       "--config_exp", str(path), "--fname", "toy", "--version", "run1", "--score"]
            completed = subprocess.run(command, cwd=REPO, env=process_env,
                                       capture_output=True, text=True, timeout=120)
            assert completed.returncode == 0, completed.stdout + completed.stderr
            assert all(file.read_bytes() == contents for file, contents in saved.items())
            assert Path(p["scores_path"]).is_file()
            report = json.loads(Path(p["metrics_path"]).read_text())
            if entry in ("carla_ae.py", "carla_vae.py"):
                assert report["selection"]["source"] == "validation loss"
                assert Path(report["selection"]["weights"]).name == "best_validation_loss.pth.tar"
            assert Path(p["calibration_path"]).is_file()
            configurations[entry] = (path, saved)
            print(f"{entry} --score: same config, reports written, training weights unchanged", flush=True)

        # Either two-phase entry can also score its shared phase-one config.
        path, saved = configurations["lewm.py"]
        for entry in ("lewm_reconstruction.py", "lewm_cross_attention.py"):
            command = [str(REPO / "venv/bin/python"), str(REPO / entry), "--config_env", str(env),
                       "--config_exp", str(path), "--fname", "toy", "--version", "run1", "--score"]
            completed = subprocess.run(command, cwd=REPO, env=process_env,
                                       capture_output=True, text=True, timeout=120)
            assert completed.returncode == 0, completed.stdout + completed.stderr
            assert all(file.read_bytes() == contents for file, contents in saved.items())
        assert not list((REPO / "configs/lewm_encoder").glob("*/reconstruction/score.yml"))
    print("All five score CLIs and shared phase-one scoring without retraining: OK")


if __name__ == "__main__":
    main()
