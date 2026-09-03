"""Component factories for the JEPA pipeline.

The single wiring convention: every construction is keyed by config
names through registries (models.BACKBONE_REGISTRY, predictor and
anti-collapse registries in models.lewm, criteria below). Legacy contrastive
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
    from models.lewm import LeWMModel

    built = get_backbone(p["backbone"], **p["model_kwargs"])
    return LeWMModel(
        encoder=built["model"],
        predictor=p.get("predictor", "tcn"),
        horizons=p.get("horizons", 2),
        predictor_hidden=p.get("predictor_hidden", None),
        anti_collapse=p.get("anti_collapse", "none"),
        ema_momentum=p.get("ema_momentum", 0.99925),
        codebook_kwargs=p.get("codebook_kwargs", None),
        target_norm=p.get("target_norm", None),
        use_projector=p.get("use_projector", False),
        projector_hidden=p.get("projector_hidden", None),
        action_dim=p.get("action_dim", 0),
    )


def _criterion_combined_aux(kw):
    from losses.combined import CombinedAuxCriterion
    from losses.detection import BoxLoss
    from losses.reconstruction import ReconLoss

    kw = dict(kw)
    return CombinedAuxCriterion(
        recon=ReconLoss(**kw.pop("recon_kwargs", {})) if kw.get("recon_kwargs") is not None else None,
        box=BoxLoss(**kw.pop("box_kwargs", {})) if kw.get("box_kwargs") is not None else None,
        w_rec=kw.pop("w_rec", 1.0), w_box=kw.pop("w_box", 1.0))


# Single map shared by all criterion branches (one place to add a tactic).
CRITERION_BUILDERS = {
    "jepa": "losses.jepa_losses:JEPALoss",
    "dense_pred": "losses.prediction:DensePartLoss",
    "recon": "losses.reconstruction:ReconLoss",
    "box": "losses.detection:BoxLoss",
    "metric": "losses.metric:MetricLoss",
    "energy": "losses.alignment:EnergyLoss",
    "viewkl": "losses.alignment:ViewKLLoss",
}


def get_criterion(p):
    """Build a registered criterion by config name."""
    import importlib

    name = p["criterion"]
    if name == "combined_aux":
        return _criterion_combined_aux(p["criterion_kwargs"])
    if name not in CRITERION_BUILDERS:
        raise ValueError("Invalid criterion {}".format(name))
    module_name, class_name = CRITERION_BUILDERS[name].split(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(**p["criterion_kwargs"])


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
