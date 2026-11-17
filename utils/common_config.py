"""Factories shared by LeWM pretraining and reconstruction."""
import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR, ConstantLR, SequentialLR

from utils.collate import collate_custom
from losses import RecCALoss, ReconL1Loss, LeWMLoss
from data import PSM, SMD, TSB


CRITERION_BUILDERS = {
    "lewm_rec_ca": RecCALoss,
    "recon_l1": ReconL1Loss,
    "lewm": LeWMLoss,
}

DATASET_BUILDER = {
    "smd": SMD,
    "psm": PSM,
    "tsb": TSB,
}

def get_criterion(p):
    """Build a criterion selected by the steering or reconstruction config."""

    name = p["criterion"]
    if name not in CRITERION_BUILDERS:
        raise ValueError("Invalid criterion {}".format(name))
    return CRITERION_BUILDERS[name](**p)

def get_dataset(p):
    joint = p.get("fname", None) == "all"
    train_dataset: object
    val_dataset: object
    db_name = p.pop("name").lower()
    val_fraction = p.pop("val_fraction", 0.1)
    if db_name in ("smd", "tsb") and joint:
        train_dataset = ...
        val_dataset = ...
    else:
        train_dataset = DATASET_BUILDER[db_name](**p)
        val_dataset = DATASET_BUILDER[db_name].validation_split(train_dataset, val_fraction)
    return train_dataset, val_dataset

def get_test_dataset(p, train_dataset):
    joint = p.get("fname", None) == "all"
    db_name = p.pop("name").lower()
    if db_name in ("smd", "tsb") and joint:
        return ...
    else:
        return DATASET_BUILDER[db_name].test_split(train_dataset)
        
def get_jepa_datasets(p):
    """Train/validation datasets for JEPA stages.

    ``stage_a.corpus: single`` (official protocol) trains on one machine;
    ``joint`` pools all SMD machines' train splits with per-machine
    normalization respected. PSM and synthetic are always their own corpus.
    Validation uses the train-side tail, or the full train series for short splits.
    """
    from data.jepa_dataset import JEPADataset, JEPACorpusDataset

    joint = p.get("fname", None) == "all"
    train_dataset: object
    val_dataset: object
    if p["train_db_name"] == "smd" and joint:
        train_dataset = JEPACorpusDataset(p)
        val_dataset = JEPACorpusDataset.validation_split(train_dataset, p)
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
        # A short-series fallback may contain fewer windows than one batch.
        drop_last=len(dataset) >= p["batch_size"],
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


def get_optimizer(p, model):
    if p["optimizer"].lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), **p["optimizer_kwargs"])

    elif p["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **p["optimizer_kwargs"])
    elif p["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), **p["optimizer_kwargs"])
    else:
        raise ValueError("Invalid optimizer {}".format(p["optimizer"]))

    return optimizer


def get_scheduler(p, optimizer, epoch):
    scheduler_name = p.pop("name")
    end_epoch = epoch
    warmup_epochs = p.get("lr_warmup_epochs", 0)
    warmup_start_factor = p.get("warmupa_start_factor", 0.1)
    warmup_end_factor = p.get("warmupa_end_factor", 1)
    warmup = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=warmup_end_factor, total_iters=warmup_epochs)

    end_epoch -= warmup_epochs
    if scheduler_name == "cosine":
        eta_min = p["lr_eta_min"]
        scheduler = CosineAnnealingLR(optimizer, T_max=end_epoch, eta_min=eta_min)
    elif scheduler_name == "cosine_restart":
        eta_min = p["lr_eta_min"]
        cycle_period = p["T_period"]
        cycle_period_mul = p.get("T_mul", 1)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cycle_period, T_mult=cycle_period_mul, eta_min=eta_min)
    elif scheduler_name == "step":
        scheduler = MultiStepLR(optimizer, milestones=p["lr_decay_epochs"], gamma=p["lr_decay_rate"])
    elif scheduler_name == "constant":
        scheduler = ConstantLR(optimizer, factor=1, total_iters=end_epoch)
    elif scheduler_name == "linear":
        scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=end_epoch)
    else:
        raise ValueError("Invalid learning rate schedule {}".format(scheduler_name))
    return SequentialLR(optimizer, [warmup, scheduler], milestones=[warmup_epochs])
