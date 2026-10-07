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

from models.builders import get_lewm_model

from utils.common_config import (
    get_criterion,
    get_jepa_datasets,
    get_optimizer,
    get_scheduler,
    get_train_dataloader,
    get_val_dataloader,
)
from utils.config import create_config, entry_overrides
from utils.masking_steered import SubAnomalyMaskCollator
from utils.trainer import Trainer
from utils.utils import Logger


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
    logger = Logger(p["version"], verbose=2, file_path=p["jepa_dir"],
                    use_tensorboard=True, delete_files=False)
    domain = p.get("predictor_kwargs", {}).get("domain", "frequency")
    with_aux = bool(p.get("aux_kwargs", {}).get("with_aux", True))
    steering = bool(p.get("predictor_kwargs", {}).get("time_steering", True))
    label = ("time annotation steering" if steering else
             "auxiliary without FiLM" if with_aux else "no auxiliary head")
    logger.log(f"LeWM {domain} predictor ({label}), stage "
               f"'{p.get('stage', 'pretrain')}' --> ")
    logger.log_hyperparams(p)
    return logger


def _masking_collator(p):
    masking = p.get("stage_a", {}).get("masking", {})
    mode = str(masking.get("mode", "none")).lower()
    if mode in ("subanomaly", "sub_anomaly", "injected"):
        kwargs = {k: v for k, v in masking.items() if k != "mode"}
        return SubAnomalyMaskCollator(**kwargs)
    return None


class _GraphWrapper(torch.nn.Module):
    """Flattens LeWM outputs for TensorBoard graph logging only."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        graph = {"latent": out["latents"]["L0"],
                 "predicted": out["predicted"]["L0"]}
        if out["mask_logits"] is not None:
            graph["mask_logits"] = out["mask_logits"]
        return graph


def _build_run(p, device):
    model = get_lewm_model(p)
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
                      collator=_masking_collator(p), amp=p.get("amp", False),
                      val_collator=_masking_collator(p))
    start_epoch, best_val_loss = Trainer.resume(p, model, optimizer, scheduler, logger)
    best_val_loss = trainer.fit(train_loader, val_loader, start_epoch, best_val_loss)
    logger.log(f"LeWM pretraining finished; best val loss {best_val_loss:.6f}")
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
        logger.log("Adaptation mode 'frozen': encoder frozen (aux+predictor train)")
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
    return score_with_model(p, device, get_lewm_model, logger, input_resolution=True)


STAGES = {
    "pretrain": run_pretrain,
    "pretext": run_pretrain,
    "adapt": run_adapt,
    "score": run_score,
}


def main(args, update_dictionary=None):
    p = create_config(args.config_env, args.config_exp, args.fname, args.version,
                      update_dictionary=entry_overrides(args, update_dictionary))
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
