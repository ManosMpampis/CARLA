"""Train or score AE, VAE, or LeWM using one experiment config."""

import argparse
import json
from pathlib import Path

import torch

from data.jepa_dataset import JEPADataset, JEPACorpusDataset
from models import get_model
from utils.common_config import (
    get_criterion, get_device, get_jepa_datasets, get_optimizer, get_scheduler,
    get_train_dataloader, get_val_dataloader, set_seed,
)
from utils.config import create_config, entry_overrides
from utils.evaluation_schedule import evaluation_interval
from utils.experiment_suite import score_config
from utils.reconstruction_baselines import aggregation_modes, evaluate_on_test
from utils.trainer import Trainer
from utils.utils import Logger
from utils.randomness import isolated_rng, isolated_module_rng


def run_contract(config):
    """Semantic runtime options; placement and epoch extensions may change."""
    fields = ("framework", "train_db_name", "dataset_collection", "fname", "machine_files",
              "wsz", "stride", "seed", "val_fraction", "criterion", "criterion_kwargs",
              "batch_size", "optimizer", "optimizer_kwargs", "score_mode", "calibration_quantile",
              "scheduler", "scheduler_kwargs", "amp", "score_tau", "evaluation",
              "validation_seed", "aggregate_output_strategy", "tsb_train_end", "tsb_evaluation_scope")
    return {name: config[name] for name in fields if name in config}


@torch.no_grad()
def check_model_loss(model, criterion, batch, device):
    """Exercise the configured output and loss contract before training."""
    with isolated_rng(), isolated_module_rng(criterion):
        return _check_model_loss(model, criterion, batch, device)


def _check_model_loss(model, criterion, batch, device):
    x = Trainer._to_model_input(batch["ts"], device)
    action = batch.get("action")
    was_training = model.training
    model.eval()
    try:
        outputs = model(x) if action is None else model(x, action=action.to(device))
        loss = criterion(outputs)["loss"]
        if "latents" not in outputs:
            raise ValueError("model output must contain latents for Trainer")
    except KeyError as exc:
        raise ValueError(f"model and loss require the output field {exc}") from exc
    finally:
        model.train(was_training)
    if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
        raise ValueError("criterion must return a finite scalar loss")


def run(config):
    config = dict(config)
    framework = config["framework"]
    stage = config.get("stage", "train")
    if stage not in ("train", "pretrain", "score"):
        raise ValueError("stage must be train, pretrain, or score")
    aggregation_modes(config)
    set_seed(int(config.get("seed", 4)))
    device = get_device(config)

    train_data, val_data = get_jepa_datasets(config)
    if not len(train_data):
        raise ValueError("training must contain at least one window")
    config["validation_available"] = bool(len(val_data))
    config["_feature_schema"] = list(train_data.feature_schema)
    config["_preprocessing"] = train_data.preprocessing
    config["_run_contract"] = run_contract(config)
    input_channels = train_data[0]["ts"].shape[-1]
    interval = evaluation_interval(config)
    test_data = (JEPADataset(config, train=False) if (stage == "score" or interval)
                 and not isinstance(train_data, JEPACorpusDataset) else None)
    from utils.model_contract import prepare_feature_source_config, initialize_feature_source
    checkpoint = Path(config["jepa_checkpoint"])
    own_state = (Path(config.get("score_checkpoint") or config["jepa_model"])
                 if stage == "score" else checkpoint)
    saved_state = torch.load(own_state, map_location="cpu", weights_only=False) if own_state.is_file() else None
    config = prepare_feature_source_config(config, train_data.feature_schema,
                                          resume_checkpoint=saved_state)
    model = get_model(config, input_channels=input_channels).to(device)
    if stage == "score":
        return score_config(config, framework, model=model, datasets=(train_data, test_data))

    train_loader = get_train_dataloader(config, train_data)
    val_loader = get_val_dataloader(config, val_data)
    if not checkpoint.is_file() and not config.get("initialize_from"):
        initialize_feature_source(model, config, train_data.feature_schema)

    criterion = get_criterion(config).to(device)
    with isolated_rng(config.get("validation_seed", int(config.get("seed", 4)) + 100003)):
        check_model_loss(model, criterion, next(iter(val_loader if len(val_data) else train_loader)), device)
    config["objective_weights"] = {name: float(getattr(criterion, name)) for name in
                                   ("w_pred", "w_aux", "w_recon", "lambda_sigreg", "lambda_sigreg_tgt", "beta")
                                   if hasattr(criterion, name)}
    import yaml
    directory = Path(config["jepa_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    plain_config = json.loads(json.dumps(config, default=str))
    (directory / "resolved_config.yml").write_text(yaml.safe_dump(plain_config, sort_keys=False))
    (directory / "architecture.txt").write_text(str(model) + "\n")
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    logger = Logger(config["version"], verbose=3, file_path=config["jepa_dir"], delete_files=False)
    try:
        logger.log(f"{framework}: {len(train_data)} training windows, {len(val_data)} validation windows")
        if not config["validation_available"]:
            logger.log("Validation unavailable: train and calibrate on all training rows; primary selection uses training loss")
        trainer = Trainer(config, model, criterion, optimizer, scheduler, device, logger,
                          amp=bool(config.get("amp", False)) and device.type == "cuda")
        start_epoch, best = Trainer.resume(config, model, optimizer, scheduler, logger)

        def evaluate(epoch):
            return evaluate_on_test(model, config, device, logger, epoch,
                                    datasets=(train_data, test_data))

        best = trainer.fit(train_loader, val_loader, start_epoch, best,
                           eval_every=interval, eval_fn=evaluate if interval else None)
        logger.log(f"Best {'validation' if config['validation_available'] else 'training'} loss {best:.6f}")
        if interval:
            score_config(config, framework, model=model, datasets=(train_data, test_data))
        return best
    finally:
        logger.finalize()


def main(args):
    from utils.jobs import plan_jobs, run_jobs
    plan = plan_jobs(args.config_exp, args.config_env,
                     dataset=getattr(args, "dataset", None), fname=getattr(args, "fname", None),
                     file_list=getattr(args, "file_list", None),
                     dataset_root=getattr(args, "dataset_root", None), version=args.version,
                     output_root=getattr(args, "output_root", None), overrides=entry_overrides(args))
    return run_jobs(plan, jobs=getattr(args, "jobs", None))


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_env", default="configs/env.yml")
    parser.add_argument("--config_exp", required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--fname")
    selection.add_argument("--file-list")
    parser.add_argument("--dataset", choices=("psm", "smd", "tsb-ad-u", "tsb-ad-m", "synthetic"))
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-root")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--version")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--score_checkpoint")
    outcome = main(parser.parse_args(argv))
    return 1 if any(row["status"] == "failed" for row in outcome["jobs"]) else 0


if __name__ == "__main__":
    raise SystemExit(cli())
