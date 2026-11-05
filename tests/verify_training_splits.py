"""Train-side fallback and full-series calibration for SMD, PSM and TSB-AD."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

import yaml

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.jepa_dataset import JEPADataset, JEPACorpusDataset
from utils import reconstruction_baselines as baselines
from utils.common_config import get_jepa_datasets, get_train_dataloader
from utils.tsb_ad_benchmark import adapt_geometry


class Logger:
    def scalar_summary(self, *args):
        pass

    def log(self, *args):
        pass


class Scores:
    def eval(self):
        return self

    def score(self, x):
        return {"fused": x.abs().mean(dim=1)}


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="training-splits-") as temporary:
        root = Path(temporary)
        raw = np.column_stack((np.arange(64), np.arange(64) ** 2)).astype(np.float32)
        labels = np.zeros(64, dtype=int)
        labels[32:40] = 1
        for split in ("train", "test", "test_label"):
            (root / split).mkdir()
        for name, values in (("train", raw), ("test", raw * 2), ("test_label", labels)):
            np.savetxt(root / name / "machine-1-1.txt", values, delimiter=",")
        for split, values in (("train", raw), ("test", raw * 2)):
            pd.DataFrame({"timestamp_(min)": np.arange(64), "a": values[:, 0],
                          "b": values[:, 1]}).to_csv(root / f"{split}.csv", index=False)
        pd.DataFrame({"time": np.arange(64), "label": labels}).to_csv(root / "test_label.csv", index=False)
        pd.DataFrame({"a": np.r_[raw[:, 0], raw[:, 0] * 2],
                      "b": np.r_[raw[:, 1], raw[:, 1] * 2],
                      "Label": np.r_[np.zeros(64, dtype=int), labels]}).to_csv(
                          root / "toy_tr_64_1st_96.csv", index=False)
        normalized = ((raw - raw.mean(axis=0)) / raw.std(axis=0)).astype(np.float32)
        for source in ("smd", "psm", "tsb_ad"):
            config = dict(train_db_name=source, dataset_root=str(root),
                          fname="toy_tr_64_1st_96.csv" if source == "tsb_ad" else "machine-1-1.txt",
                          wsz=16, stride=8, batch_size=4, val_fraction=0.25)
            for window, fraction, expected_train, expected_val in (
                (16, 0.25, 48, 16),  # exactly one held-out window
                (17, 0.25, 64, 64),  # validation too short
                (17, 0.75, 64, 64),  # optimization split too short
                (64, 0.25, 64, 64),  # exactly one full-series window
            ):
                train, val = get_jepa_datasets(dict(config, wsz=window, val_fraction=fraction))
                assert len(train.series) == expected_train and len(val.series) == expected_val, source
                assert len(train) > 0 and len(val) > 0
                np.testing.assert_allclose(train.full_train_series, normalized, atol=1e-6)
                if expected_val == 64:
                    np.testing.assert_array_equal(train.series, val.series)
            # Monitoring must score the whole training series even with a valid holdout.
            original_score = baselines.score_both
            scored_lengths = []

            def capture(model, series, *args, **kwargs):
                scored_lengths.append(len(series))
                return original_score(model, series, *args, **kwargs)

            # Metrics are irrelevant to the calibration-input contract here.
            with patch.object(baselines, "score_both", side_effect=capture), \
                 patch.object(baselines, "evaluate_from_scores", return_value={}), \
                 patch.object(baselines, "_evaluate_protocol", return_value={}), \
                 patch.object(baselines, "_log_evaluation"), \
                 patch.object(baselines, "record_test_evaluation"):
                baselines.evaluate_on_test(Scores(), config, "cpu", Logger(), 1)
            assert scored_lengths == [64, 64, 64, 64], (source, scored_lengths)
            # One complete window must still yield a training batch.
            one_window = JEPADataset(dict(config, wsz=64), train=True)
            loader = get_train_dataloader(dict(config, num_workers=0), one_window)
            assert len(loader) == 1 and next(iter(loader))["ts"].shape[0] == 1

            # Exercise real optimization and checkpoint selection on the fallback.
            repo = Path(__file__).resolve().parents[1]
            arm_config = yaml.safe_load((repo / "configs/baselines/ae.yml").read_text())
            arm_config.update(config, wsz=64, device="cpu", epochs=1, eval_every=0,
                              num_workers=0, amp=False, scheduler="constant",
                              scheduler_kwargs={"lr_warmup_epochs": 0, "warmupa_start_factor": 1.0},
                              model_kwargs={"in_channels": 2, "enc_channels": [4, 8],
                                            "enc_strides": [2, 2], "dropout": 0.0})
            config_path = root / f"{source}.yml"
            config_path.write_text(yaml.safe_dump(arm_config))
            env_path = root / "env.yml"
            env_path.write_text(yaml.safe_dump({"root_dir": str(root / "outputs")}))
            best = baselines.train_arm("ae", SimpleNamespace(
                config_env=str(env_path), config_exp=str(config_path),
                fname=config["fname"], version="short"))
            assert np.isfinite(best["validation_loss"]), (source, best)
            try:
                JEPADataset(dict(config, wsz=65), train=True)
            except ValueError as exc:
                assert "need at least" in str(exc)
            else:
                raise AssertionError("a training series shorter than one window was accepted")

        # Joint SMD must apply the fallback independently for each machine.
        np.savetxt(root / "train/machine-1-2.txt", raw[:32], delimiter=",")
        corpus = JEPACorpusDataset(dict(config, train_db_name="smd", wsz=16,
                                       joint_cache_dir=str(root / "cache")))
        validation = JEPACorpusDataset.validation_split(corpus)
        assert len(corpus) == 5 + 3 and len(validation) == 1 + 3
        assert validation[1]["meta"]["start_idx"] == 0
        geometry = adapt_geometry(dict(config, wsz=32, val_fraction=0.1), 64, "strict")
        assert geometry["wsz"] == 32 and geometry["batch_size"] == 4
    print("Training splits: SMD/PSM/TSB-AD fallback and full-training calibration OK")


if __name__ == "__main__":
    main()
