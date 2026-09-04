"""Full-LeWM training entry: trunk plus attached heads in one run.

Builds the assembly declared by the experiment YAML (`heads:` section
selects recon_aux / box_aux / h1..h4; `head_criterion_kwargs` weights the
terms) and trains it through the shared Trainer, so checkpoint/resume,
TensorBoard, and mixed-precision behave exactly like the trunk-only entry.

Scoring reuses the trunk-only entry: the best trunk weights are exported
alongside the full state and load straight into it for calibration.
"""
import argparse
import os

import torch

from utils.common_config import (
    get_full_criterion,
    get_full_model,
    get_jepa_datasets,
    get_optimizer,
    get_scheduler,
    get_train_dataloader,
    get_val_dataloader,
)
from utils.config import create_config
from utils.masking import MaskingCollator
from utils.trainer import Trainer
from utils.utils import Logger


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch for deterministic runs."""
    import random

    import numpy as np

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
                    use_tensorboard=True, delete_files=True)
    logger.log(f"CARLA full-LeWM stage '{p.get('stage', 'pretrain')}' --> ")
    logger.log_hyperparams(p)
    return logger


def _masking_collator(p):
    masking = p.get("stage_a", {}).get("masking", {})
    if masking.get("mode", "none") != "block":
        return None
    kwargs = {k: v for k, v in masking.items() if k != "mode"}
    return MaskingCollator(**kwargs)


class _GraphWrapper(torch.nn.Module):
    """Trunk-only view of the full model for TensorBoard graph logging."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model.trunk(x)
        return {f"latent/{n}": z for n, z in out["latents"].items()}


def _build_run(p, device):
    model = get_full_model(p)
    criterion = get_full_criterion(p).to(device)
    optimizer = get_optimizer(p, model)
    scheduler = get_scheduler(p, optimizer)
    return model, criterion, optimizer, scheduler


def run_pretrain(p, device):
    """Train trunk + heads from scratch; export full and trunk weights."""
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

    trunk_path = os.path.join(p["jepa_dir"], "trunk.pth.tar")
    torch.save(model.trunk_state_dict(), trunk_path)
    logger.log(f"Full-LeWM finished; best val loss {best_val_loss:.6f}; "
               f"trunk weights for the scoring entry at {trunk_path}")
    logger.finalize()


def run_adapt(p, device):
    """Adapt full weights to the target machine (frozen trunk default)."""
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
        for param in model.trunk.encoder.parameters():
            param.requires_grad = False
        model.trunk.encoder_frozen = True
        logger.log("Adaptation mode 'frozen': trunk encoder frozen")
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
    trunk_path = os.path.join(p["jepa_dir"], "trunk.pth.tar")
    torch.save(model.trunk_state_dict(), trunk_path)
    logger.log(f"Adaptation finished; best val loss {best_val_loss:.6f}")
    logger.finalize()


STAGES = {"pretrain": run_pretrain, "pretext": run_pretrain, "adapt": run_adapt}


def main(args, update_dictionary={}):
    """Dispatch on the YAML stage key, mirroring the trunk-only entry."""
    p = create_config(args.config_env, args.config_exp, args.fname, args.version,
                      update_dictionary=update_dictionary)
    set_seed(int(p.get("seed", 4)))
    stage = str(p.get("stage", "pretrain")).lower()
    if stage not in STAGES:
        raise ValueError(f"Invalid stage {stage}; expected one of {sorted(STAGES)}")
    STAGES[stage](p, _device(p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-LeWM training harness")
    parser.add_argument("--config_env", help="Config file for the environment")
    parser.add_argument("--config_exp", help="Config file for the experiment")
    parser.add_argument("--fname", help="File name of the dataset machine", default="")
    parser.add_argument("--version", help="Experiment version", type=str)
    args = parser.parse_args()
    main(args)
