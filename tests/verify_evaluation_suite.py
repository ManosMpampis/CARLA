"""Checkpoint-only suite: real scoring of all runners on temporary PSM data."""

import base64
import copy
import re
import csv
import json
from pathlib import Path
import sys
import tempfile

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.verify_psm_suite import make_psm
from utils.evaluation_suite import build_evaluation_plan, run_evaluation_suite
from utils.experiment_suite import build_plan, _build_model
from utils.trainer import Trainer


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    with tempfile.TemporaryDirectory(prefix="evaluation-suite-") as temporary:
        directory = Path(temporary)
        make_psm(directory)
        env = directory / "env.yml"
        env.write_text(yaml.safe_dump({"root_dir": str(directory / "training-results")}))
        entries = []
        for framework, name, runner, config in (
            ("ae", "default", "ae", "baselines/ae.yml"),
            ("vae", "default", "vae", "baselines/vae.yml"),
            ("lewm_encoder", "time", "lewm", "lewm_encoder/time/phase1.yml"),
            ("reconstruction", "default", "recon", "lewm_encoder/time/reconstruction/default.yml"),
            ("cross_attention", "observed_input", "cross_attention", "lewm_encoder/time/cross_attention/observed_input.yml"),
        ):
            entry = dict(framework=framework, experiment_name=name, runner=runner,
                         config=str(REPO / "configs" / config))
            if runner in ("recon", "cross_attention"):
                entry["pretrained_experiment"] = "lewm_encoder/time"
            entries.append(entry)
        common = dict(train_db_name="psm", val_db_name="psm", fname="psm",
                      device="cpu", wsz=64, stride=32, val_fraction=0.25,
                      batch_size=4, score_batch_size=16, eval_window_size=8,
                      dataset_root=str(directory / "PSM"), decision_trace=False,
                      model_kwargs=dict(in_channels=25, enc_channels=[4, 8],
                                        enc_strides=[2, 2], dropout=0.0),
                      predictor_kwargs=dict(stem_channels=4, neck_widths=[4, 4, 4]),
                      cross_attention_kwargs=dict(qk_channels=4, value_channels=4,
                                                  context_crop=[0, 8], target_crop=[8, 16],
                                                  dropout=0.0))
        training = directory / "experiments.yml"
        training.write_text(yaml.safe_dump(dict(common=common, experiments=entries)))
        plan, _ = build_plan(training, env, "default")
        evaluation_entries = []
        original_bytes = {}
        for exp in plan:
            weights_dir = directory / "weights" / exp.key
            weights_dir.mkdir(parents=True)
            model = _build_model(exp, exp.config)
            checkpoints = []
            for index in range(2 if exp.runner == "cross_attention" else 1):
                path = weights_dir / f"epoch_{index}.pth.tar"
                # Both raw state dicts and full Trainer checkpoints are supported.
                payload = model.state_dict() if index == 0 else {"model": model.state_dict(), "epoch": index}
                torch.save(payload, path)
                original_bytes[path] = path.read_bytes()
                checkpoints.append(dict(name=f"epoch_{index}", path=str(path)))
                if exp.runner == "cross_attention" and index == 1:
                    checkpoints[-1]["overrides"] = {"decision_trace": {"max_windows": 2}}
            (weights_dir / "resolved_config.yml").write_text(yaml.safe_dump(exp.config))
            cases = [dict(name="small", input_window=64, stride=32)]
            if exp.runner == "cross_attention":
                cases[0].update(context_crop=[0, 4], target_crop=[4, 16])
                cases.append(dict(name="large", input_window=128, stride=32,
                                  context_crop=[0, 12], target_crop=[12, 32]))
                # Strict training task mismatch must still fail independently.
                wrong = copy.deepcopy(exp.config)
                wrong["cross_attention_kwargs"]["target_crop"] = [8, 32]
                try:
                    Trainer.load_weights(str(weights_dir / "epoch_0.pth.tar"), _build_model(exp, wrong))
                except ValueError as error:
                    assert "task differs" in str(error)
                else:
                    raise AssertionError("training task mismatch was accepted")
            evaluation_entries.append(dict(experiment=exp.key, checkpoints=checkpoints,
                                           evaluations=cases))
        manifest = directory / "evaluation.yml"
        document = dict(name="test", experiments_manifest="experiments.yml",
                        common={"inference_benchmark": {"warmup_batches": 1, "measured_batches": 2}},
                        output_dir="evaluation-results", experiments=evaluation_entries)
        manifest.write_text(yaml.safe_dump(document))
        jobs, output, version = build_evaluation_plan(manifest, env)
        assert len(jobs) == 8
        assert len({job.output_dir for job in jobs}) == 8
        assert not output.exists() and not (directory / "training-results").exists()
        rows = run_evaluation_suite(jobs, output, env, version)
        assert all(row["status"] == "completed" for row in rows), rows
        assert not (directory / "training-results").exists()
        assert all(path.read_bytes() == content for path, content in original_bytes.items())
        for job in jobs:
            report = json.loads((job.output_dir / "metrics.json").read_text())
            assert {"timeseries", "window", "non_overlap_window"} == report["evaluation"].keys()
            assert report["selection"]["source"] == "configured checkpoint"
            assert (job.output_dir / "Test_timeseries_scores.csv").is_file()
            assert not (job.output_dir / "scores.npz").exists()
            assert (job.output_dir / "calibration.json").is_file()
            speed = report["inference_speed"]
            assert speed["measured_batches"] == 2
            assert speed["ms_per_window"] > 0 and speed["windows_per_second"] > 0
            assert abs(speed["ms_per_window"] * speed["windows_per_second"] - 1000) < 1e-6
            assert not (job.output_dir / "suite_evaluation").exists()
            for section in ("Test_window", "Test_non_overlap", "Test_timeseries"):
                assert (job.output_dir / f"{section}_metrics.csv").is_file()
                html = (job.output_dir / f"{section}.html").read_text()
                assert len(html.encode()) < 500_000
                encoded = re.search(r"const encodedTrace = '([^']+)'", html).group(1)
                trace = json.loads(base64.b64decode(encoded))
                assert trace['series_length'] == report['coverage']['total_timesteps']
                windows = trace['checking_windows']
                assert all(b-a == report['coverage']['input_window'] for a,b,_,_ in windows)
                assert all(d-c == report['coverage']['output_window'] for _,_,c,d in windows)
                if section == 'Test_timeseries':
                    assert trace['score_count'] == report['coverage']['scored_timesteps']
                else:
                    assert trace['score_count'] == len(windows)
            if job.experiment.runner == "cross_attention":
                large = job.case_name == "large"
                assert report["coverage"]["output_window"] == (20 if large else 12)
                assert report["coverage"]["first_scored_timestep"] == (12 if large else 4)
        # Failure isolation: missing checkpoint does not stop the next job.
        failed = copy.deepcopy(jobs[:2])
        failed[0].experiment.config["score_checkpoint"] = str(directory / "missing.pth.tar")
        rows = run_evaluation_suite(failed, output, env, version)
        assert [row["status"] for row in rows] == ["failed", "completed"]
        assert len(json.loads((output / "summary.json").read_text())) == 8
        assert (output / "Test_timeseries_summary.csv").is_file()
        with (output / "Test_timeseries_summary.csv").open() as stream:
            summaries = list(csv.DictReader(stream))
        assert all(float(row["inference_ms_per_window"]) > 0 for row in summaries
                   if row["status"] == "completed")
        # All-experiment sweeps reuse weights with raw-timestep asymmetric crops.
        sweep = dict(document, experiments="all", evaluations=[dict(name="half", input_window=32)],
                     cross_attention_evaluations=[dict(name="asymmetric", input_window=48,
                                                       context_length=32, target_length=16)])
        # Select standard checkpoints from the explicit fixtures above.
        for entry in entries:
            config = next(exp for exp in plan if exp.runner == entry["runner"])
            from utils.experiment_suite import checkpoint_path
            selected = checkpoint_path(config, directory / "training-results", "default")
            selected.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_build_model(config, config.config).state_dict(), selected)
            (selected.parent / "resolved_config.yml").write_text(yaml.safe_dump(config.config))
        sweep["output_dir"] = "sweep-results"
        manifest.write_text(yaml.safe_dump(sweep))
        sweep_jobs, sweep_output, sweep_version = build_evaluation_plan(manifest, env)
        assert len(sweep_jobs) == 6
        sweep_rows = run_evaluation_suite(sweep_jobs, sweep_output, env, sweep_version)
        assert all(row["status"] == "completed" for row in sweep_rows), sweep_rows
        asymmetric = next(row for row in sweep_rows if row["case"] == "asymmetric")
        assert asymmetric["evaluation_settings"]["context_length"] == 32
        assert asymmetric["evaluation_settings"]["target_length"] == 16
        assert all(path.read_bytes() == content for path, content in original_bytes.items())
        # Duplicate names must never overwrite another evaluation job.
        document["experiments"].append(document["experiments"][0])
        manifest.write_text(yaml.safe_dump(document))
        try:
            build_evaluation_plan(manifest, env)
        except ValueError as error:
            assert "duplicate evaluation job" in str(error)
        else:
            raise AssertionError("duplicate evaluation job accepted")
    print("verify_evaluation_suite: OK (all runners, multiple checkpoints/crops, strict loading, isolation)")


if __name__ == "__main__":
    main()
