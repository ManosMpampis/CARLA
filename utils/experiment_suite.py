"""Compatibility adapters and primary-checkpoint scoring for native training.

Dataset fan-out, selection preflight and concurrency live in utils.jobs. Every
training adapter below delegates to train.run; deleted legacy runners are not
imported or reintroduced.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from utils.config import create_config, experiment_base_dir


@dataclass
class Experiment:
    framework: str
    name: str
    runner: str
    config_path: Path
    config: dict
    dependency: str | None = None
    run_version: str | None = None

    @property
    def key(self):
        return f"{self.framework}/{self.name}"


def experiment_dir(exp, root_dir, version):
    base = Path(experiment_base_dir(root_dir, exp.config, exp.config["fname"], exp.run_version or version))
    tag = exp.config.get("tag_jepa")
    return base / (f"jepa_{tag}" if tag else "jepa")


def checkpoint_path(exp, root_dir, version):
    return experiment_dir(exp, root_dir, version) / "model.pth.tar"


def train_experiment(exp, env_path, version):
    from train import run
    config = create_config(env_path, exp.config_path, exp.config.get("fname"),
                           exp.run_version or version, update_dictionary=exp.config)
    return run(config)


def score_experiment(exp, env_path, version, *, fname=None, output_dir=None):
    config = create_config(env_path, exp.config_path, fname or exp.config.get("fname"),
                           exp.run_version or version,
                           update_dictionary={**exp.config, "stage": "score"}, output_dir=output_dir)
    return score_config(config, exp)


def score_config(p, exp=None, *, model=None, datasets=None):
    """Score the primary best checkpoint with the same native training model."""
    import torch
    from models import get_model
    from utils.common_config import get_device, get_jepa_datasets
    from utils.model_contract import prepare_feature_source_config
    from utils.run_evaluation import evaluate_run
    from utils.trainer import Trainer
    p = dict(p)
    if datasets is None:
        from data.jepa_dataset import JEPADataset, JEPACorpusDataset
        train_data, _ = get_jepa_datasets(p)
        test_data = None if isinstance(train_data, JEPACorpusDataset) else JEPADataset(p, train=False)
        datasets = train_data, test_data
    weights = Path(p.get("score_checkpoint") or p["jepa_model"])
    if not weights.is_file():
        raise FileNotFoundError(f"primary selected checkpoint missing: {weights}")
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    p = prepare_feature_source_config(p, datasets[0].feature_schema, resume_checkpoint=checkpoint)
    p["_feature_schema"] = list(datasets[0].feature_schema)
    p["_preprocessing"] = datasets[0].preprocessing
    device = get_device(p)
    if model is None:
        model = get_model(p, input_channels=len(datasets[0].feature_schema)).to(device)
    Trainer.validate_run_contract(p, checkpoint)
    Trainer.load_weights(weights, model, strict=True)
    report = evaluate_run(model, p, device, datasets=datasets, write=True)
    report["selection"] = {"weights": str(weights), **checkpoint.get("selection", {})}
    path = Path(p["jepa_dir"]) / "metrics.json"
    path.write_text(json.dumps(report, indent=2))
    return report
