"""Dataset-native paths, SMD machine isolation, training, evaluation, and resume."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.jepa_dataset import make_synthetic_series
from utils.config import DEFAULT_VERSION, experiment_base_dir
from utils.evaluation_suite import build_evaluation_plan, run_evaluation_suite
from utils.experiment_suite import (build_plan, checkpoint_path, experiment_dir,
                                    load_suite_manifest, resume_latest, run_suite)
from experiments_smd_all import cli


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    with tempfile.TemporaryDirectory(prefix="dataset-suites-") as temporary:
        directory = Path(temporary)
        root = directory / "outputs"
        env = directory / "env.yml"
        env.write_text(yaml.safe_dump({"root_dir": str(root)}))
        psm, _ = build_plan(REPO / "configs/psm/experiments.yml", env, DEFAULT_VERSION)
        smd, _ = build_plan(REPO / "configs/smd/experiments.yml", env, DEFAULT_VERSION,
                            fname="machine-1-1.txt")
        assert {exp.key for exp in psm} == {exp.key for exp in smd}
        assert all(exp.config["train_db_name"] == "smd" and exp.config["model_kwargs"]["in_channels"] == 38 for exp in smd)
        assert all(exp.config["train_db_name"] == "psm" and exp.config["model_kwargs"]["in_channels"] == 25 for exp in psm)
        assert experiment_dir(psm[0], root, DEFAULT_VERSION) == root / "psm/ae/default/jepa_ae"
        assert experiment_dir(smd[0], root, DEFAULT_VERSION) == root / "smd/ae/default/machine-1-1.txt/jepa_ae"
        assert not root.exists()
        data_root = directory / "SMD"
        for split in ("train", "test", "test_label"):
            (data_root / split).mkdir(parents=True)
        machines = ["machine-1-1.txt", "machine-1-2.txt"]
        for index, machine in enumerate(machines):
            train, _ = make_synthetic_series(512, 38, index + 4)
            test, _ = make_synthetic_series(256, 38, index + 14)
            labels = np.zeros(256, dtype=int)
            labels[160:190] = 1
            test[160:190, :3] += 6
            for split, values in (("train", train), ("test", test), ("test_label", labels)):
                np.savetxt(data_root / split / machine, values, delimiter=",")
        document = load_suite_manifest(REPO / "configs/smd/experiments.yml")
        selected = {"ae/default", "vae/default", "lewm_encoder/time", "lewm_encoder/frequency",
                    "lewm_encoder/time/reconstruction/default",
                    "lewm_encoder/time/cross_attention/observed_input"}
        configs = {str(exp.config_path) for exp in smd if exp.key in selected}
        document["experiments"] = [entry for entry in document["experiments"] if entry["config"] in configs]
        document["common"].update(
            dataset_root=str(data_root), fname=machines[0], epochs=1, eval_every=0, device="cpu", wsz=64,
            stride=32, batch_size=4, score_batch_size=16, eval_window_size=8,
            val_fraction=0.25, decision_trace=False,
            model_kwargs=dict(in_channels=38, enc_channels=[4, 8], enc_strides=[2, 2], dropout=0.0),
            predictor_kwargs=dict(stem_channels=4, neck_widths=[4, 4, 4], n_fft=16, hop_length=4, win_length=16),
            cross_attention_kwargs=dict(qk_channels=4, value_channels=4, dropout=0.0))
        manifest = directory / "experiments.yml"
        manifest.write_text(yaml.safe_dump(document))
        plan, _ = build_plan(manifest, env, DEFAULT_VERSION)
        assert len(plan) == 6 and {exp.key for exp in plan} == selected
        rows = run_suite(plan, root, env, DEFAULT_VERSION)
        assert all(row["status"] == "completed" for row in rows), rows
        assert not (root / "psm").exists()
        for exp in plan:
            report = json.loads((experiment_dir(exp, root, DEFAULT_VERSION) / "metrics.json").read_text())
            assert report["dataset"] == "smd" and report["fname"] == machines[0]
            if exp.dependency:
                source = next(item for item in plan if item.key == exp.dependency)
                assert exp.config["pretrained_from"] == str(checkpoint_path(source, root, DEFAULT_VERSION))
        assert (root / "smd/summaries" / machines[0] / "summary.csv").is_file()
        second, _ = build_plan(manifest, env, DEFAULT_VERSION, experiments=["ae/default"], fname=machines[1])
        rows = run_suite(second, root, env, DEFAULT_VERSION)
        assert rows[0]["status"] == "completed", rows
        assert (root / "smd/summaries" / machines[1] / "summary.csv").is_file()
        assert checkpoint_path(second[0], root, DEFAULT_VERSION) != checkpoint_path(plan[0], root, DEFAULT_VERSION)
        # Stable and named run discovery retain the correct machine identity.
        named = copy.deepcopy(second[0])
        state_dir = experiment_dir(named, root, "named")
        state_dir.mkdir(parents=True)
        (state_dir / "last.pth.tar").touch()
        resume_latest(named, root, DEFAULT_VERSION)
        assert named.run_version == "named"
        # Omitted encoders resolve from dataset + machine, never another dataset.
        child_doc = copy.deepcopy(document)
        child_doc["experiments"] = [entry for entry in document["experiments"] if entry["runner"] == "cross_attention"]
        child_manifest = directory / "child.yml"
        child_manifest.write_text(yaml.safe_dump(child_doc))
        child, _ = build_plan(child_manifest, env, DEFAULT_VERSION)
        assert child[0].config["pretrained_from"] == plan[-1].config["pretrained_from"]
        # Checkpoint evaluation defaults also derive their dataset and machine.
        evaluation = directory / "evaluation.yml"
        evaluation.write_text(yaml.safe_dump(dict(
            name="smd-check", experiments_manifest="experiments.yml",
            experiments=[dict(experiment="ae/default", checkpoints=[dict(name="selected")])])) )
        jobs, output, version = build_evaluation_plan(evaluation, env)
        assert output == root / "smd/evaluations/smd-check" / machines[0]
        evaluated = run_evaluation_suite(jobs, output, env, version)
        assert evaluated[0]["status"] == "completed", evaluated
        # All-machine CLI discovery is read-only and includes both files.
        with patch.object(sys, "argv", ["suite", "--manifest", str(manifest), "--config-env", str(env),
                                         "--all-machines", "--dry-run", "--frameworks", "ae"]), \
             patch("utils.suite_cli.build_plan", wraps=build_plan) as planner:
            cli()
            assert [call.kwargs["fname"] for call in planner.call_args_list] == machines
        # Standalone configs also omit redundant single-series identifiers.
        assert Path(experiment_base_dir(root, {"train_db_name": "psm"}, "psm", DEFAULT_VERSION)) == root / "psm"
    print("verify_dataset_suites: OK (SMD all runners, two machines, paths, handoffs, discovery and evaluation)")


if __name__ == "__main__":
    main()
