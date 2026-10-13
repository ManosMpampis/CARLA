"""Native AE/VAE benchmark training, weights-only fine-tuning, and LR resume.

Run with ./venv/bin/python tests/verify_tsb_ad_training.py.
All artifacts and synthetic data stay in a temporary directory.
"""

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

from data.jepa_dataset import JEPADataset
from utils import reconstruction_baselines as baselines
from utils.config import create_config


def main():
    torch.set_num_threads(1)
    original_init = JEPADataset.__init__
    original_epoch = baselines._run_epoch

    def train_data_only(self, config, train, transform=None):
        assert train, "benchmark training attempted to construct a test dataset"
        return original_init(self, config, train, transform)

    with tempfile.TemporaryDirectory(prefix="verify-tsb-training-") as temporary:
        root = Path(temporary)
        env = root / "env.yml"
        env.write_text(yaml.safe_dump({"root_dir": str(root / "results")}))
        paths, args_by_arm = {}, {}
        for arm in ("ae", "vae"):
            config = yaml.safe_load((REPO / f"configs/baselines/smd_{arm}.yml").read_text())
            config.update(
                train_db_name="synthetic", val_db_name="synthetic",
                synthetic_kwargs={"n_steps": 256, "test_n_steps": 64, "n_channels": 2},
                model_kwargs={"in_channels": 2, "enc_channels": [4, 8],
                              "enc_strides": [2, 2], "norm": "batch", "dropout": 0.0},
                recon_kwargs={"dropout": 0.0}, wsz=16, stride=8,
                val_fraction=0.25, device="cpu", epochs=1, batch_size=4,
                num_workers=0, seed=4, scheduler="constant",
                scheduler_kwargs={"lr_warmup_epochs": 0, "warmupa_start_factor": 1.0},
                benchmark_protocol="tsb_ad",
                # The protocol must block these even if a native YAML enables them.
                eval_every=1, decision_trace={"max_windows": 2},
            )
            paths[arm] = root / f"{arm}.yml"
            paths[arm].write_text(yaml.safe_dump(config))
            args = SimpleNamespace(config_env=str(env), config_exp=str(paths[arm]),
                                   fname="toy", version="benchmark")
            args_by_arm[arm] = args
            with patch.object(JEPADataset, "__init__", train_data_only):
                baselines.train_arm(arm, args)
            resolved = create_config(env, paths[arm], "toy", "benchmark")
            directory = Path(resolved["jepa_dir"])
            selected = torch.load(directory / "best_validation_loss.pth.tar",
                                  map_location="cpu", weights_only=False)
            last = torch.load(directory / "last.pth.tar", map_location="cpu", weights_only=False)
            assert selected["selection"]["source"] == "validation"
            assert selected["evaluation"] is None and last["evaluation"] is None
            assert last["benchmark_protocol"] == "tsb_ad"
            assert not list(directory.glob("best_test_*"))
            assert not (directory / "test_evaluations.jsonl").exists()
            assert not (directory / "decision_trace.html").exists()
            epochs = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
            assert len(epochs) == 1 and epochs[0]["evaluation"] is None
            text = (directory / "log.txt").read_text()
            assert "TSB-AD protocol" in text and "epoch 1" in text and "hyperparams" in text
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            events = EventAccumulator(str(directory / "tensorboard")).Reload()
            tags = events.Tags()["scalars"]
            assert "train/loss" in tags and "validation/loss" in tags
            assert not any(tag.startswith("test/") for tag in tags)

        source_config = create_config(env, paths["ae"], "toy", "benchmark")
        source = Path(source_config["jepa_dir"]) / "best_validation_loss.pth.tar"
        expected = torch.load(source, map_location="cpu", weights_only=False)["model"]
        fresh_args = SimpleNamespace(**vars(args_by_arm["ae"]))
        fresh_args.version = "finetune"
        seen_initialization = []

        def inspect_initialization(model, loader, optimizer, device, beta, writer, epoch, training):
            if training and not seen_initialization:
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value.cpu(), expected[name])
                assert optimizer.param_groups[0]["lr"] == 0.0007
                assert not optimizer.state, "weights-only initialization restored optimizer state"
                seen_initialization.append(epoch)
            return original_epoch(model, loader, optimizer, device, beta, writer, epoch, training)

        with patch.object(JEPADataset, "__init__", train_data_only), \
                patch.object(baselines, "_run_epoch", inspect_initialization):
            baselines.train_arm("ae", fresh_args, {
                "initialize_from": str(source), "optimizer_kwargs": {"lr": 0.0007}})
        assert seen_initialization == [1]
        finetune_config = create_config(env, paths["ae"], "toy", "finetune")
        state_path = Path(finetune_config["jepa_dir"]) / "last.pth.tar"
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        assert state["initialize_from"] == str(source)
        resume_epochs = []

        def inspect_resume(model, loader, optimizer, device, beta, writer, epoch, training):
            if training:
                assert optimizer.param_groups[0]["lr"] == 0.0001
                assert optimizer.state, "resume discarded optimizer state"
                resume_epochs.append(epoch)
            return original_epoch(model, loader, optimizer, device, beta, writer, epoch, training)

        with patch.object(JEPADataset, "__init__", train_data_only), \
                patch.object(baselines, "_run_epoch", inspect_resume):
            baselines.train_arm("ae", fresh_args, {
                "epochs": 3, "initialize_from": str(root / "does-not-exist.pth"),
                "resume_learning_rate": 0.0001})
        assert resume_epochs == [2, 3]
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        assert state["epoch"] == 3 and state["initialize_from"] == str(source)
        assert state["optimizer"]["param_groups"][0]["lr"] == 0.0001
        print("Native TSB-AD training, validation-only selection, warm start, and LR resume verified")


if __name__ == "__main__":
    main()
