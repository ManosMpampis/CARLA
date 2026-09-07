"""True-LeWM training entry: two-stream input masking, one shared encoder.

Mirrors carla_jepa.py stage-for-stage (pretrain/adapt/score, checkpoint /
resume, TensorBoard, AMP) but builds TrueLeWMModel (models/lewm_true.py)
instead of LeWMModel: the encoder sees mask(X) and X in two forward passes
and an action-conditioned predictor bridges the gap. Scoring reuses the
shared engine (utils.reporting.score_with_model), so calibration, scores,
and metrics land in the same files as every other arm.
"""
import argparse
import os
import random

import numpy as np
import torch

from utils.common_config import (
    get_criterion,
    get_jepa_datasets,
    get_optimizer,
    get_scheduler,
    get_train_dataloader,
    get_val_dataloader,
)
from utils.config import create_config
from utils.masking_true import InputBlockMaskCollator
from utils.trainer import Trainer
from utils.utils import Logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_true_model(p):
    """Shared PyramidEncoder + per-level action-conditioned predictors."""
    from models import get_backbone
    from models.lewm_true import TrueLeWMModel

    built = get_backbone("jepa_pyramid", **p["model_kwargs"])
    return TrueLeWMModel(
        encoder=built["model"],
        action_dim=p.get("action_dim", 16),
        predictor_kwargs=p.get("predictor_kwargs", None),
    )


def _device(p):
    want = str(p.get("device", "cpu")).lower()
    if want.startswith("cuda") and not torch.cuda.is_available():
        want = "cpu"
    return torch.device(want)


def _make_logger(p):
    destructive = str(p.get("stage", "pretrain")).lower() != "score"
    logger = Logger(p["version"], verbose=2, file_path=p["jepa_dir"],
                    use_tensorboard=True, delete_files=destructive)
    logger.log(f"CARLA true-LeWM stage '{p.get('stage', 'pretrain')}' --> ")
    logger.log_hyperparams(p)
    return logger


def _masking_collator(p):
    masking = p.get("stage_a", {}).get("masking", {})
    if masking.get("mode", "none") != "block_input":
        return None
    kwargs = {k: v for k, v in masking.items() if k != "mode"}
    return InputBlockMaskCollator(**kwargs)


class _GraphWrapper(torch.nn.Module):
    """Flattens TrueLeWMModel outputs for TensorBoard graph logging only."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        flat = {}
        for name, z in out["latents"].items():
            flat[f"latent/{name}"] = z
            flat[f"predicted/{name}"] = out["predicted"][name][:, 0]
            flat[f"context/{name}"] = out["context"][name]
        return flat


def _build_run(p, device):
    model = get_true_model(p)
    criterion = get_criterion(p).to(device)
    optimizer = get_optimizer(p, model)
    scheduler = get_scheduler(p, optimizer)
    return model, criterion, optimizer, scheduler


def run_pretrain(p, device):
    logger = _make_logger(p)
    model, criterion, optimizer, scheduler = _build_run(p, device)

    in_channels = p["model_kwargs"]["in_channels"]
    try:
        graph_model = _GraphWrapper(model).to(device)
        logger.add_graph(graph_model,
                         torch.rand((1, in_channels, p["wsz"]), device=device))
    except Exception as exc:
        logger.warn(f"TensorBoard graph logging skipped: {exc}")
    model = model.to(device)

    train_dataset, val_dataset = get_jepa_datasets(p)
    train_loader = get_train_dataloader(p, train_dataset)
    val_loader = get_val_dataloader(p, val_dataset)
    logger.log(f"Dataset contains {len(train_dataset)}/{len(val_dataset)} "
               f"train/val samples")

    trainer = Trainer(p, model, criterion, optimizer, scheduler, device, logger,
                      collator=_masking_collator(p), amp=p.get("amp", False))
    start_epoch, best_val_loss = Trainer.resume(p, model, optimizer, scheduler, logger)
    best_val_loss = trainer.fit(train_loader, val_loader, start_epoch, best_val_loss)
    logger.log(f"True-LeWM pretraining finished; best val loss {best_val_loss:.6f}")
    logger.finalize()


def run_adapt(p, device):
    logger = _make_logger(p)
    model, criterion, optimizer, scheduler = _build_run(p, device)

    source = p.get("pretrained_from")
    if not source or not os.path.exists(source):
        raise FileNotFoundError(
            f"adaptation requires 'pretrained_from' checkpoint; got {source}"
        )
    Trainer.load_weights(source, model, logger, strict=True)

    mode = p.get("stage_b", {}).get("mode", "frozen")
    if mode == "frozen":
        for param in model.encoder.parameters():
            param.requires_grad = False
        model.encoder_frozen = True
        logger.log("Adaptation mode 'frozen': encoder frozen")
    elif mode == "finetune":
        logger.log("Adaptation mode 'finetune': all parameters update")
    else:
        raise ValueError(f"Invalid stage_b.mode {mode}")

    model = model.to(device)
    optimizer = get_optimizer(p, model)
    scheduler = get_scheduler(p, optimizer)

    train_dataset, val_dataset = get_jepa_datasets(p)
    train_loader = get_train_dataloader(p, train_dataset)
    val_loader = get_val_dataloader(p, val_dataset)
    trainer = Trainer(p, model, criterion, optimizer, scheduler, device, logger,
                      collator=None, amp=p.get("amp", False))
    start_epoch, best_val_loss = Trainer.resume(p, model, optimizer, scheduler, logger)
    best_val_loss = trainer.fit(train_loader, val_loader, start_epoch, best_val_loss)
    logger.log(f"Adaptation finished; best val loss {best_val_loss:.6f}")
    logger.finalize()


@torch.no_grad()
def run_score(p, device):
    """Score stage: identical report flow through the shared engine."""
    from utils.reporting import score_with_model

    logger = _make_logger(p)
    score_with_model(p, device, get_true_model, logger)


STAGES = {
    "pretrain": run_pretrain,
    "pretext": run_pretrain,
    "adapt": run_adapt,
    "score": run_score,
}


def main(args, update_dictionary={}):
    p = create_config(args.config_env, args.config_exp, args.fname, args.version,
                      update_dictionary=update_dictionary)
    set_seed(int(p.get("seed", 4)))
    stage = str(p.get("stage", "pretrain")).lower()
    if stage not in STAGES:
        raise ValueError(f"Invalid stage {stage}; expected one of {sorted(STAGES)}")
    STAGES[stage](p, _device(p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="True-LeWM TSAD harness")
    parser.add_argument("--config_env", help="Config file for the environment")
    parser.add_argument("--config_exp", help="Config file for the experiment")
    parser.add_argument("--fname", help="File name of the dataset machine", default="")
    parser.add_argument("--version", help="Experiment version", type=str)
    args = parser.parse_args()
    main(args)
