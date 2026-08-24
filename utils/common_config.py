"""Component factories for the JEPA pipeline.

The single wiring convention: every construction is keyed by config
names through registries (models.BACKBONE_REGISTRY, predictor and
anti-collapse registries in models.jepa_core). Legacy contrastive
wiring was removed at cutover; the legacy dataset classes remain in
data/custom_dataset.py, unused but functional.
"""
import os

import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR, ConstantLR, SequentialLR

from utils.collate import collate_custom
from utils.mypath import MyPath


def get_jepa_model(p):
    """Single wiring point for JEPA arms: registry-built encoder + config
    selected predictor and anti-collapse mechanism."""
    from models import get_backbone
    from models.jepa_core import JEPAModel

    built = get_backbone(p["backbone"], **p["model_kwargs"])
    return JEPAModel(
        encoder=built["model"],
        predictor=p.get("predictor", "tcn"),
        horizons=p.get("horizons", 2),
        predictor_hidden=p.get("predictor_hidden", None),
        anti_collapse=p.get("anti_collapse", "none"),
        ema_momentum=p.get("ema_momentum", 0.99925),
        codebook_kwargs=p.get("codebook_kwargs", None),
        target_norm=p.get("target_norm", None),
    )


def get_criterion(p):
    if p["criterion"] == "jepa":
        from losses.jepa_losses import JEPALoss

        return JEPALoss(**p["criterion_kwargs"])
    raise ValueError("Invalid criterion {}".format(p["criterion"]))


def get_jepa_datasets(p):
    """Train/validation datasets for JEPA stages.

    ``stage_a.corpus: single`` (official protocol) trains on one machine;
    ``joint`` pools all SMD machines' train splits with per-machine
    normalization respected. PSM and synthetic are always their own corpus.
    Validation windows come exclusively from the train-side tail.
    """
    from data.jepa_dataset import JEPADataset, JEPACorpusDataset

    corpus = p.get("stage_a", {}).get("corpus", "single")
    train_dataset: object
    val_dataset: object
    if p["train_db_name"] == "smd" and corpus == "joint":
        machine_dir = os.path.join(MyPath.db_root_dir("smd"), "train")
        machines = sorted(f for f in os.listdir(machine_dir) if f.startswith("machine-"))
        train_dataset = JEPACorpusDataset(p, machines)
        val_dataset = JEPACorpusDataset.validation_split(train_dataset)
    else:
        train_dataset = JEPADataset(p, train=True)
        val_dataset = JEPADataset.validation_split(train_dataset, p)
    return train_dataset, val_dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=p["num_workers"],
        batch_size=p["batch_size"],
        pin_memory=False,
        collate_fn=collate_custom,
        drop_last=True,
        shuffle=True,
    )


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=p["num_workers"],
        batch_size=p["batch_size"],
        pin_memory=False,
        collate_fn=collate_custom,
        drop_last=False,
        shuffle=False,
    )


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only:  # Only weights in the cluster head will be updated
        for name, param in model.named_parameters():
            if "cluster_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
    else:
        params = model.parameters()

    if p["optimizer"].lower() == "sgd":
        optimizer = torch.optim.SGD(params, **p["optimizer_kwargs"])

    elif p["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(params, **p["optimizer_kwargs"])
    elif p["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(params, **p["optimizer_kwargs"])
    else:
        raise ValueError("Invalid optimizer {}".format(p["optimizer"]))

    return optimizer


def get_scheduler(p, optimizer):
    end_epoch = p["epochs"]
    warmup_epochs = p["scheduler_kwargs"].get("lr_warmup_epochs", 0)
    warmup_start_factor = p["scheduler_kwargs"].get("warmupa_start_factor", 0.1)
    warmup_end_factor = p["scheduler_kwargs"].get("warmupa_end_factor", 1)
    warmup = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=warmup_end_factor, total_iters=warmup_epochs)

    end_epoch -= warmup_epochs
    if p["scheduler"] == "cosine":
        eta_min = p["scheduler_kwargs"]["lr_eta_min"]
        scheduler = CosineAnnealingLR(optimizer, T_max=end_epoch, eta_min=eta_min)
    elif p["scheduler"] == "cosine_restart":
        eta_min = p["scheduler_kwargs"]["lr_eta_min"]
        cycle_period = p["scheduler_kwargs"]["T_period"]
        cycle_period_mul = p["scheduler_kwargs"].get("T_mul", 1)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cycle_period, T_mult=cycle_period_mul, eta_min=eta_min)
    elif p["scheduler"] == "step":
        scheduler = MultiStepLR(optimizer, milestones=p["scheduler_kwargs"]["lr_decay_epochs"], gamma=p["scheduler_kwargs"]["lr_decay_rate"])
    elif p["scheduler"] == "constant":
        scheduler = ConstantLR(optimizer, factor=1, total_iters=end_epoch)
    elif p["scheduler"] == "linear":
        scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=end_epoch)
    else:
        raise ValueError("Invalid learning rate schedule {}".format(p["scheduler"]))
    return SequentialLR(optimizer, [warmup, scheduler], milestones=[warmup_epochs])
