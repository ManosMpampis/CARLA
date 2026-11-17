"""LeWM training entry with selectable prediction domain and auxiliary head.

The pretrain/adapt/score stages support checkpoint resume, TensorBoard, and
AMP. LeWMModel uses injected anomalies during training: the shared encoder
processes clean and injected windows without stop-gradient. The optional
auxiliary never receives the action; the mask token conditions either predictor.
Scoring uses the shared reporting engine.
"""
import argparse
import os
import random

import numpy as np
import torch

from models.builders import get_model

from utils.common_config import (
    get_criterion,
    get_dataset,
    get_optimizer,
    get_scheduler,
    get_train_dataloader,
    get_val_dataloader,
)
from utils.config import create_config
from utils.evaluation_schedule import evaluation_interval
from utils.reconstruction_baselines import evaluate_on_test
from utils.trainer import Trainer
from utils.utils import Logger

def load_feature_extractor(source, model, logger=None):
    """Strict encoder-only transfer from either saved weights or a Trainer checkpoint."""
    if not source or not os.path.isfile(source):
        raise FileNotFoundError(f"phase2 requires pretrained_from LEWM weights; got {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload["model"] if "model" in payload else payload
    encoder_state = {key.removeprefix("encoder."): value
                     for key, value in state.items() if key.startswith("encoder.")}
    if not encoder_state:
        raise ValueError(f"no encoder weights in LEWM checkpoint {source}")
    model.encoder.load_state_dict(encoder_state, strict=True)
    if model.encoder_frozen:
        model.encoder.eval()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(p):
    want = str(p.get("device", "cpu")).lower()
    if want.startswith("cuda") and not torch.cuda.is_available():
        want = "cpu"
    return torch.device(want)


def _make_logger(p):
    logger = Logger(p["version"], verbose=3, file_path=p["jepa_dir"],
                    use_tensorboard=True, delete_files=False)
    domain = p.get("predictor_kwargs", {}).get("domain", "frequency")
    with_aux = bool(p.get("aux_kwargs", {}).get("with_aux", True))
    steering = bool(p.get("predictor_kwargs", {}).get("time_steering", True))
    label = ("time annotation steering" if steering else
             "auxiliary without FiLM" if with_aux else "no auxiliary head")
    logger.log(f"LeWM {domain} predictor ({label}), stage "
               f"'{p.get('stage', 'pretrain')}' --> ")
    return logger


def _build_run(p, device):
    model = get_model(p.get("framework"))
    criterion = get_criterion(p.get("criterion")).to(device)
    optimizer = get_optimizer(p.get("optimizer"), model)
    scheduler = get_scheduler(p.get("scheduler"), optimizer, p["epochs"])
    return model, criterion, optimizer, scheduler


def run_train(p, device):
    logger = _make_logger(p)

    train_dataset, val_dataset = get_dataset(p.get("data"))
    if train_dataset.features != p["framework"]["encoder"]["in_channels"]:
        raise UserWarning(f"Dataset has different number of feature than the input of the model"
                          f" Data:{train_dataset.features}/Model:{p["framework"]["encoder"]["in_channels"]}." 
                          f" We change the configureation of the model")
    p["framework"]["encoder"]["in_channels"] = train_dataset.features

    model, criterion, optimizer, scheduler = _build_run(p, device)

    in_channels = p["framework"]["encoder"]["in_channels"]
    try:
        logger.add_graph(model, torch.rand((1, in_channels, p["wsz"]), device=device))
    except Exception as exc:
        logger.warn(f"TensorBoard graph logging skipped: {exc}")
    model = model.to(device)

    train_loader = get_train_dataloader(p, train_dataset)
    val_loader = get_val_dataloader(p, val_dataset)
    logger.log(f"Dataset contains {len(train_dataset)}/{len(val_dataset)} "
               f"train/val samples")

    trainer = Trainer(p, model, criterion, optimizer, scheduler, device, logger, amp=p.get("amp", {"enabled": False, "dtype": torch.float32}))
    start_epoch, best_val_loss = Trainer.resume(p, model, optimizer, scheduler, logger)
    interval = evaluation_interval(p)
    best_val_loss = trainer.fit(
        train_loader, val_loader, start_epoch, best_val_loss,
        eval_every=interval,
        eval_fn=(lambda step: evaluate_on_test(
            model, p, device, logger, step, input_resolution=True))
        if interval > 0 else None)
    logger.log(f"LeWM pretraining finished; best val loss {best_val_loss:.6f}")
    logger.finalize()


@torch.no_grad()
def run_score(p, device):
    """Export the same three-mode report as every experiment runner."""
    from utils.experiment_suite import score_config
    return score_config(p, "lewm")


STAGES = {
    "pretrain": run_train,
    "score": run_score,
}


def main(args, update_dictionary=None):
    p = create_config(args.config_env, args.config_exp, args.fname, args.version,
                      update_dictionary=update_dictionary)
    set_seed(int(p.get("seed", 4)))
    stage = str(p.get("stage", "pretrain")).lower()
    if stage not in STAGES:
        raise ValueError(f"Invalid stage {stage}; expected one of {sorted(STAGES)}")
    return STAGES[stage](p, _device(p))


def cli():
    parser = argparse.ArgumentParser(description="LeWM time-series experiment harness")
    parser.add_argument("--config_env", help="Config file for the environment")
    parser.add_argument("--config_exp", help="Config file for the experiment")
    parser.add_argument("--fname", help="File name of the dataset machine", default="")
    parser.add_argument("--version", help="Experiment version", type=str)
    parser.add_argument("--score", action="store_true", help="score saved weights using this experiment config")
    parser.add_argument("--stage", choices=sorted(STAGES))
    parser.add_argument("--score_checkpoint", help="optional weights path; defaults to this run's weights")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
