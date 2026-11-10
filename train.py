"""Train or score AE, VAE, or LeWM using one experiment config."""

import argparse
from pathlib import Path

import torch

from data.jepa_dataset import JEPADataset
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


@torch.no_grad()
def check_model_loss(model, criterion, batch, device):
    """Exercise the configured output and loss contract before training."""
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
    framework = config["framework"]
    stage = config.get("stage", "train")
    if stage not in ("train", "pretrain", "score"):
        raise ValueError("stage must be train, pretrain, or score")
    aggregation_modes(config)
    set_seed(int(config.get("seed", 4)))
    device = get_device(config)

    train_data, val_data = get_jepa_datasets(config)
    if not len(train_data) or not len(val_data):
        raise ValueError("training and validation must each contain at least one window")
    input_channels = train_data[0]["ts"].shape[-1]
    interval = evaluation_interval(config)
    test_data = JEPADataset(config, train=False) if stage == "score" or interval else None
    model = get_model(config, input_channels=input_channels).to(device)
    if stage == "score":
        return score_config(config, framework, model=model, datasets=(train_data, test_data))

    train_loader = get_train_dataloader(config, train_data)
    val_loader = get_val_dataloader(config, val_data)
    backbone = config.get("lewm_modules", {}).get("backbone") or {}
    checkpoint = Path(config["jepa_checkpoint"])
    if (backbone.get("pretrained_from") and not checkpoint.is_file()
            and not config.get("initialize_from")):
        Trainer.initialize_backbone(model, backbone["pretrained_from"],
                                    backbone.get("checkpoint_prefix", "encoder."))

    criterion = get_criterion(config).to(device)
    check_model_loss(model, criterion, next(iter(val_loader)), device)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    logger = Logger(config["version"], verbose=3, file_path=config["jepa_dir"], delete_files=False)
    try:
        logger.log(f"{framework}: {len(train_data)} training windows, {len(val_data)} validation windows")
        trainer = Trainer(config, model, criterion, optimizer, scheduler, device, logger,
                          amp=bool(config.get("amp", False)) and device.type == "cuda")
        start_epoch, best = Trainer.resume(config, model, optimizer, scheduler, logger)

        def evaluate(epoch):
            return evaluate_on_test(model, config, device, logger, epoch,
                                    datasets=(train_data, test_data))

        best = trainer.fit(train_loader, val_loader, start_epoch, best,
                           eval_every=interval, eval_fn=evaluate if interval else None)
        logger.log(f"Best validation loss {best:.6f}")
        return best
    finally:
        logger.finalize()


def main(args):
    config = create_config(args.config_env, args.config_exp, args.fname, args.version,
                           update_dictionary=entry_overrides(args))
    return run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_env", default="configs/env.yml")
    parser.add_argument("--config_exp", required=True)
    parser.add_argument("--fname")
    parser.add_argument("--version")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--score_checkpoint")
    main(parser.parse_args())
