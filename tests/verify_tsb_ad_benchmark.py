"""Exercise every inherited model through the native TSB-AD benchmark CLI.

Run ./venv/bin/python tests/verify_tsb_ad_benchmark.py. The small CPU runs use
temporary synthetic CSVs and require no benchmark download or upstream package.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.tsb_ad import read_tsb_series
from models.builders import get_cross_attention_model
from utils.config import merge_config
from utils.experiment_suite import checkpoint_path, experiment_dir
from utils.trainer import Trainer
from utils.tsb_ad_benchmark import aligned_scores, build_benchmark_plan, record_run


def rejected(action, message):
    try:
        action()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected rejection containing {message!r}")


def write_series(path, length=208):
    steps = np.arange(length, dtype=np.float64)
    data = np.column_stack((np.sin(steps * 0.1), np.cos(steps * 0.07)))
    labels = np.zeros(length, dtype=np.int64)
    labels[-24:-12] = 1
    data[-24:-12] += 4
    pd.DataFrame({"first": data[:, 0], "second": data[:, 1],
                  "Label": labels}).to_csv(path, index=False)


def load_state(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    torch.set_num_threads(1)
    np.testing.assert_array_equal(aligned_scores({
        "scores": np.array([np.nan, np.nan, 2, np.nan, np.nan, 8, np.nan]),
        "cover_counts": np.array([0, 0, 1, 0, 0, 1, 0]),
    }), [2, 2, 2, 4, 6, 8, 8])
    rejected(lambda: aligned_scores({"scores": np.array([np.inf]),
                                      "cover_counts": np.array([1])}), "nonfinite")
    rejected(lambda: aligned_scores({"scores": np.array([0.0]),
                                      "cover_counts": np.array([0])}), "empty")

    with tempfile.TemporaryDirectory(prefix="verify-tsb-benchmark-") as temporary:
        workspace = Path(temporary)
        dataset = workspace / "data"
        dataset.mkdir()
        path = dataset / "001_Toy_id_1_Facility_tr_160_1st_184.csv"
        write_series(path)
        sample = read_tsb_series(path)
        output = workspace / "results"
        env = workspace / "env.yml"
        env.write_text(yaml.safe_dump({"root_dir": str(output)}))
        overrides = {
            "model_kwargs": {"enc_channels": [4, 8], "enc_strides": [2, 2], "dropout": 0.0},
            "predictor_kwargs": {"stem_channels": 4, "neck_widths": [8, 8, 8],
                                 "n_fft": 16, "hop_length": 4, "win_length": 16},
            "aux_kwargs": {"aux_channels": [4, 4, 4]},
            "cross_attention_kwargs": {"qk_channels": 4, "value_channels": 4,
                                       "num_heads": 1, "dropout": 0.0},
            "recon_kwargs": {"dropout": 0.0},
            "scheduler_kwargs": {"lr_warmup_epochs": 0, "warmupa_start_factor": 1.0},
            "wsz": 32, "stride": 16, "val_fraction": 0.2, "batch_size": 4,
            "score_batch_size": 8, "num_workers": 0, "amp": False,
        }
        override_path = workspace / "overrides.yml"
        override_path.write_text(yaml.safe_dump(overrides))
        manifest = REPO / "configs/tsb_ad/experiments.yml"

        def cli(version="suite", *, extra=(), override=override_path, expected=0, filename=path.name):
            command = [sys.executable, str(REPO / "benchmark_tsb_ad.py"),
                       "--dataset-root", str(dataset), "--files", filename,
                       "--config-env", str(env), "--output-root", str(output),
                       "--manifest", str(manifest), "--overrides", str(override),
                       "--version", version, "--epochs", "1", "--device", "cpu",
                       "--cpu-threads", "1", *extra]
            process = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                                     env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                          "MPLCONFIGDIR": str(workspace / "matplotlib")})
            assert process.returncode == expected, (
                f"Command failed ({process.returncode}): {' '.join(command)}\n"
                + process.stdout[-18000:] + process.stderr[-18000:])
            return json.loads((output / "benchmark_runs" / version / "summary.json").read_text())

        plan = build_benchmark_plan(manifest, env, "suite", sample, output,
                                    overrides=overrides, epochs=1, device="cpu")
        assert len(plan) == 36, f"expected all 36 inherited variants, got {len(plan)}"
        assert {exp.runner for exp in plan} == {"ae", "vae", "lewm", "recon", "cross_attention"}
        rows = cli()
        assert len(rows) == len(plan) and all(row["status"] == "completed" for row in rows)
        row_by_key = {row["experiment"]: row for row in rows}
        original_scores = {}
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        for exp in plan:
            directory = experiment_dir(exp, output, "suite")
            row = row_by_key[exp.key]
            score = np.load(row["score_path"], allow_pickle=False)
            original_scores[exp.key] = score.copy()
            assert score.shape == sample.labels.shape and np.isfinite(score).all()
            np.testing.assert_array_equal(np.load(row["upstream_score_path"], allow_pickle=False), score)
            report = json.loads((directory / "metrics.json").read_text())
            assert report["metric_status"] == "deferred" and "tsb_ad" not in report
            assert Path(report["checkpoint"]) == checkpoint_path(exp, output, "suite").resolve()
            assert report["calibrated"]["source"] == "declared_train_prefix_validation_tail"
            assert report["score_length"] == len(sample.data)
            assert report["coverage"]["covered"] + report["coverage"]["uncovered"] == len(score)
            with np.load(directory / "scores.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["scores"], score)
                np.testing.assert_array_equal(archive["labels"], sample.labels)
                assert archive["cover_counts"].shape == score.shape
            for filename in ("log.txt", "benchmark.txt", "benchmark_run.json", "resolved_config.yml",
                             "decision_trace.html", "decision_trace.json", "calibration.json"):
                assert (directory / filename).is_file(), (exp.key, filename)
            assert "hyperparams" in (directory / "log.txt").read_text()
            events = EventAccumulator(str(directory / "tensorboard")).Reload()
            tags = events.Tags()["scalars"]
            assert any("train" in tag.lower() for tag in tags), (exp.key, tags)
            assert any(tag.startswith("validation/") for tag in tags), (exp.key, tags)
            assert any(tag.startswith("TSB-AD/calibrated/") for tag in tags), (exp.key, tags)
            assert not any(tag.startswith("test/") for tag in tags), (exp.key, tags)
            assert not list(directory.glob("best_test_*"))
            assert not (directory / "model_best_eval.pth.tar").exists()
            assert not (directory / "test_evaluations.jsonl").exists()
            if exp.runner in ("ae", "vae"):
                selected = load_state(checkpoint_path(exp, output, "suite"))
                assert selected["selection"]["source"] == "validation"
                assert selected["evaluation"] is None
                assert load_state(directory / "last.pth.tar")["epoch"] == 1
            else:
                state = load_state(directory / "checkpoint.pth.tar")
                assert state["next_epoch"] == 1 and np.isfinite(state["best_val_loss"])
            if exp.dependency:
                source = next(item for item in plan if item.key == exp.dependency)
                assert Path(exp.config["pretrained_from"]) == checkpoint_path(source, output, "suite")
                assert Path(exp.config["pretrained_from"]).is_file()

        scored = cli(extra=("--score-only",))
        assert all(row["status"] == "completed" for row in scored)
        for row in scored:
            np.testing.assert_array_equal(np.load(row["score_path"], allow_pickle=False),
                                          original_scores[row["experiment"]])

        resumed = cli(extra=("--experiments", "ae/default", "lewm_encoder/time",
                             "--epochs", "2", "--learning-rate", "0.0001"))
        assert len(resumed) == 36 and all(row["status"] == "completed" for row in resumed)
        for key, filename in (("ae/default", "last.pth.tar"),
                              ("lewm_encoder/time", "checkpoint.pth.tar")):
            exp = next(item for item in plan if item.key == key)
            state = load_state(experiment_dir(exp, output, "suite") / filename)
            assert state.get("next_epoch", state["epoch"]) == 2
            assert state["optimizer"]["param_groups"][0]["lr"] == 0.0001

        cross = next(exp for exp in plan if exp.runner == "cross_attention"
                     and exp.config["phase1_experiment"] == "time" and exp.name == "observed_features")
        source_weights = checkpoint_path(cross, output, "suite")
        unfrozen = merge_config(overrides, {"cross_attention_kwargs": {"freeze_encoder": False}})
        unfrozen_path = workspace / "unfrozen.yml"
        unfrozen_path.write_text(yaml.safe_dump(unfrozen))
        warm = cli("warm", override=unfrozen_path, extra=("--experiments", cross.key,
                    "--initialize-from", str(source_weights)))
        assert len(warm) == 1 and warm[0]["status"] == "completed", warm
        warm_plan = build_benchmark_plan(manifest, env, "warm", sample, output, overrides=unfrozen,
                                         epochs=1, device="cpu", experiments=[cross.key])
        warm_target = next(exp for exp in warm_plan if exp.key == cross.key)
        warm_dir = experiment_dir(warm_target, output, "warm")
        assert not Path(warm_target.config["pretrained_from"]).exists(), "warm start retrained phase one"
        assert load_state(warm_dir / "checkpoint.pth.tar")["initialize_from"] == str(source_weights)
        model = get_cross_attention_model(warm_target.config)
        Trainer.initialize_weights(source_weights, model)
        assert not model.encoder_frozen and all(parameter.requires_grad for parameter in model.encoder.parameters())
        changed_task = merge_config(warm_target.config, {"cross_attention_kwargs": {"attention_dropout": 0.2}})
        rejected(lambda: Trainer.initialize_weights(source_weights, get_cross_attention_model(changed_task)),
                 "checkpoint task differs")
        # A later resume need not repeat --initialize-from; its original source
        # remains recorded independently of the latest runtime configuration.
        record_run(warm_target, output, "warm", sample, score_only=True)
        metadata = json.loads((warm_dir / "benchmark_run.json").read_text())
        assert metadata["initialization_source"] == str(source_weights)
        mismatch = replace(warm_target, config=merge_config(warm_target.config, {"score_mode": "mse"}))
        rejected(lambda: record_run(mismatch, output, "warm", sample), "configuration or input changed")
        original_csv = path.read_bytes()
        try:
            changed = pd.read_csv(path)
            changed.loc[len(changed) - 1, "first"] += 1
            changed.to_csv(path, index=False)
            rejected(lambda: record_run(warm_target, output, "warm", read_tsb_series(path)),
                     "configuration or input changed")
        finally:
            path.write_bytes(original_csv)

        short_path = dataset / "002_Short_id_1_Facility_tr_80_1st_100.csv"
        write_series(short_path, length=128)
        short_rows = cli("short", filename=short_path.name, expected=1,
                         extra=("--experiments", "ae/default", "lewm_encoder/frequency", "--no-trace"))
        short_status = {row["experiment"]: row["status"] for row in short_rows}
        assert short_status == {"ae/default": "completed", "lewm_encoder/frequency": "failed"}, short_rows
        odd_path = dataset / "003_Odd_id_1_Facility_tr_140_1st_184.csv"
        write_series(odd_path)
        odd_plan = build_benchmark_plan(manifest, env, "odd", read_tsb_series(odd_path), output,
                                        overrides=overrides, epochs=1, device="cpu", experiments=[cross.key])
        odd_target = next(exp for exp in odd_plan if exp.key == cross.key)
        assert odd_target.config["wsz"] == 24
        get_cross_attention_model(odd_target.config).crop_ranges(odd_target.config["wsz"])

    print("TSB-AD benchmark verified: all 36 variants, score alignment/replay, resume, warm start, and provenance")


if __name__ == "__main__":
    main()
