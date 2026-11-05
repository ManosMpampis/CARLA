"""TSB prefix training, suffix testing, scheduled native reports, and resume."""

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

from data.jepa_dataset import JEPADataset
from utils.experiment_suite import load_suite_manifest


def main():
    with tempfile.TemporaryDirectory(prefix="tsb-native-") as temporary:
        root = Path(temporary)
        dataset = root / "data"
        dataset.mkdir()
        name = "001_Toy_tr_160_1st_200.csv"
        t = np.arange(256)
        values = np.column_stack((np.sin(t / 8), np.cos(t / 11)))
        labels = np.zeros(256, dtype=int)
        labels[200:220] = 1
        values[200:220] += 4
        pd.DataFrame({"a": values[:, 0], "b": values[:, 1], "Label": labels}).to_csv(dataset / name, index=False)
        config = dict(train_db_name="tsb_ad", dataset_root=str(dataset), fname=name,
                      wsz=32, stride=16, val_fraction=0.25)
        train = JEPADataset(config, train=True)
        test = JEPADataset(config, train=False)
        assert len(train.series) == 120 and len(train.val_series) == 40
        assert len(test.series) == 96, "TSB testing still includes the training prefix"
        np.testing.assert_array_equal(test.targets, labels[160:])
        np.testing.assert_allclose(test.series, ((values[160:] - train.mean) / train.std).astype(np.float32))

        manifest = load_suite_manifest(REPO / "configs/tsb_ad/experiments.yml")
        chosen = {str(REPO / path) for path in (
            "configs/baselines/ae.yml", "configs/baselines/vae.yml",
            "configs/lewm_encoder/time/phase1.yml",
            "configs/lewm_encoder/time/reconstruction/default.yml",
            "configs/lewm_encoder/time/cross_attention/observed_input.yml")}
        manifest["experiments"] = [entry for entry in manifest["experiments"] if entry["config"] in chosen]
        assert len(manifest["experiments"]) == 5
        manifest["common"].update(
            wsz=32, stride=16, val_fraction=0.25, epochs=3, eval_every=2,
            eval_window_size=4, batch_size=4, score_batch_size=8, device="cpu",
            model_kwargs={"enc_channels": [4, 8], "enc_strides": [2, 2], "dropout": 0.0},
            predictor_kwargs={"stem_channels": 4, "neck_widths": [8, 8, 8]},
            recon_kwargs={"dropout": 0.0},
            cross_attention_kwargs={"context_crop": None, "target_crop": None,
                                    "qk_channels": 4, "value_channels": 4, "num_heads": 1},
            scheduler_kwargs={"lr_warmup_epochs": 0, "warmupa_start_factor": 1.0})
        path = root / "manifest.yml"
        path.write_text(yaml.safe_dump(manifest))
        output = root / "results"
        command = [sys.executable, str(REPO / "benchmark_tsb_ad.py"), "--manifest", str(path),
                   "--dataset-root", str(dataset), "--output-root", str(output),
                   "--version", "native", "--device", "cpu", "--cpu-threads", "1", "--no-trace"]

        def run(extra=()):
            result = subprocess.run([*command, *extra], cwd=REPO, capture_output=True, text=True,
                                    env={**os.environ, "MPLCONFIGDIR": str(root / "matplotlib")})
            assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]
            rows = json.loads((output / "benchmark_runs/native/summary.json").read_text())
            assert len(rows) == 5 and all(row["status"] == "completed" for row in rows)
            return rows

        def check(rows, expected_epochs):
            for row in rows:
                directory = Path(row["directory"])
                history = [json.loads(line) for line in (directory / "test_evaluations.jsonl").read_text().splitlines()]
                assert [item["epoch"] for item in history] == expected_epochs, (row["experiment"], history)
                for item in history:
                    assert set(item["evaluation"]) == {"window", "timeseries", "non_overlap_window"}
                    assert "f1" in item["evaluation"]["timeseries"]["calibrated"]
                for section in ("Test_window", "Test_timeseries", "Test_non_overlap"):
                    assert (directory / f"{section}.csv").exists()
                    assert section in (directory / "log.txt").read_text()
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                tags = EventAccumulator(str(directory / "tensorboard")).Reload().Tags()["scalars"]
                assert "Test_timeseries/calibrated/f1" in tags
                checkpoint = torch.load(directory / "best_validation_loss.pth.tar", map_location="cpu", weights_only=False)
                assert checkpoint["selection"]["source"] == "validation"
                assert np.load(row["score_path"]).shape == (96,)
                with np.load(row["evaluation_inputs_path"]) as inputs:
                    np.testing.assert_array_equal(inputs["labels"], labels[160:])
                report = json.loads((directory / "metrics.json").read_text())
                assert report["evaluation_start"] == 160 and report["evaluation_scope"] == "test"
                assert set(report["evaluation"]) == {"window", "timeseries", "non_overlap_window"}

        check(run(), [2, 3])
        check(run(("--epochs", "4")), [2, 3, 4])
    print("TSB native workflow: all five runners monitor test suffix before final epoch, export aligned scores, and resume")


if __name__ == "__main__":
    main()
