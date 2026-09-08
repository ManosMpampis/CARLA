"""Steered-LeWM training entry: single-scale ResNet + time aux + freq predictor.

Mirrors carla_lewm_true.py stage-for-stage (pretrain/adapt/score, checkpoint /
resume, TensorBoard, AMP) but builds SteeredFreqLeWMModel
(models/steered_lewm.py) with SubAnomalyMaskCollator
(utils/masking_steered.py): the encoder sees X_clean and X_inj in two forward
passes with no stop-grad anywhere, the auxiliary never receives the action,
and the mask feeds only the frequency stem. Scoring reuses the shared engine
(utils.reporting.score_with_model).
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


def get_steered_model(p):
    """Steered ResNet encoder + time auxiliary + frequency predictor."""
    from models import get_backbone
    from models.steered_lewm import SteeredFreqLeWMModel

    enc_kwargs = dict(p.get("model_kwargs", {}))
    built = get_backbone(p.get("backbone", "steered_resnet"), **enc_kwargs)
    aux_kwargs = dict(p.get("aux_kwargs", {}))
    pred_kwargs = dict(p.get("predictor_kwargs", {}))
    return SteeredFreqLeWMModel(
        encoder=built["model"],
        aux_channels=aux_kwargs.get("aux_channels", (32, 32, 32)),
        stem_channels=pred_kwargs.get("stem_channels", 64),
        neck_widths=tuple(pred_kwargs.get("neck_widths", (64, 64, 64))),
        n_fft=int(pred_kwargs.get("n_fft", 64)),
        hop_length=int(pred_kwargs.get("hop_length", 16)),
        win_length=int(pred_kwargs.get("win_length", 64)),
        aux_kernels=tuple(aux_kwargs.get("kernels", (7, 5, 3))),
        norm=enc_kwargs.get("norm", "batch"),
        dropout=enc_kwargs.get("dropout", True),
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
    logger.log(f"CARLA steered-LeWM stage '{p.get('stage', 'pretrain')}' --> ")
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
    """Flattens steered-model outputs for TensorBoard graph logging only."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return {"latent": out["latents"]["L0"],
                "predicted": out["predicted"]["L0"],
                "mask_logits": out["mask_logits"]}


def _build_run(p, device):
    model = get_steered_model(p)
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
    logger.log(f"Steered-LeWM pretraining finished; best val loss {best_val_loss:.6f}")
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
    score_with_model(p, device, get_steered_model, logger)


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
    parser = argparse.ArgumentParser(description="Steered-LeWM TSAD harness")
    parser.add_argument("--config_env", help="Config file for the environment")
    parser.add_argument("--config_exp", help="Config file for the experiment")
    parser.add_argument("--fname", help="File name of the dataset machine", default="")
    parser.add_argument("--version", help="Experiment version", type=str)
    args = parser.parse_args()
    main(args)
