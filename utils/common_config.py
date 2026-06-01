import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR, ConstantLR, SequentialLR
from data.augment import NoiseTransformation, SubAnomaly
from utils.collate import collate_custom


def get_criterion(p):
    if p["criterion"] == "pretext":
        from losses.losses import PretextLoss

        criterion = PretextLoss(p["batch_size"], **p["criterion_kwargs"])

    elif p["criterion"] == "classification":
        from losses.losses import ClassificationLoss

        criterion = ClassificationLoss(**p["criterion_kwargs"])
    elif p["criterion"] == "classification_e2e":
        from losses._losses import ClassificationLossE2E

        criterion = ClassificationLossE2E(**p["criterion_kwargs"])

    elif p["criterion"] == "tcl":
        from losses.tcl import TCLoss

        criterion = TCLoss(p["batch_size"], **p["criterion_kwargs"])
    elif p["criterion"] == "pretext_new":
        from losses._losses import PretextLoss

        criterion = PretextLoss(p["batch_size"], **p["criterion_kwargs"])
    elif p["criterion"] == "classification_new":
        from losses._losses import ClassificationLoss

        criterion = ClassificationLoss(**p["criterion_kwargs"])
    else:
        raise ValueError("Invalid criterion {}".format(p["criterion"]))

    return criterion


def get_feature_dimensions_backbone(p):
    if p["backbone"] == "resnet18":
        return p["res_kwargs"]["mid_channels"][-1]

    elif p["backbone"] == "resnet_ts":
        return p["res_kwargs"]["mid_channels"][-1]

    else:
        raise NotImplementedError


def get_model(p, pretrain_path=None):
    # Get backbone
    if p["backbone"] == "resnet_ts":
        from models import resnet_ts

        backbone = resnet_ts(**p["res_kwargs"])

    else:
        raise ValueError("Invalid backbone {}".format(p["backbone"]))

    # Setup
    if p["setup"] in ["pretext"]:
        from models.models import ContrastiveModel

        model = ContrastiveModel(backbone, **p["model_kwargs"])

    elif p["setup"] in ["classification"]:
        from models.models import ClusteringModel

        model = ClusteringModel(backbone, p["num_classes"], p["num_heads"])
    elif p["setup"] in ["classification_e2e"]:
        from models.models import ClusteringModel
        from models.models import ClassificationModel

        model = ClassificationModel(ClusteringModel(backbone, p["num_classes"], p["num_heads"]), p["num_classes_classificaton"])
    else:
        raise ValueError("Invalid setup {}".format(p["setup"]))

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location="cpu", weights_only=False)

        if (
            p["setup"] in ["classification", "classification_e2e"]
        ):  # Weights are supposed to be transfered from contrastive training
            missing = model.load_state_dict(state, strict=False)
            assert (
                set(missing[1])
                == {
                    "contrastive_head.1.weight",
                    "contrastive_head.1.bias",
                    "contrastive_head.3.weight",
                    "contrastive_head.3.bias",
                }
                or set(missing[1])
                == {"contrastive_head.0.weight", "contrastive_head.0.bias"}
                or set(missing[1]) == set()
            )

        else:
            raise NotImplementedError

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError(
            "Path with pre-trained weights does not exist {}".format(pretrain_path)
        )

    else:
        pass

    return model


def get_train_dataset(
    p,
    transform,
    sanomaly,
    to_augmented_dataset=False,
    to_neighbors_dataset=False,
    split=None,
    data=None,
    label=None,
):
    # Base dataset
    mean, std = 0, 0
    if p["train_db_name"] == "smd":
        from data.SMD import SMD

        dataset = SMD(
            p["fname"],
            train=True,
            transform=transform,
            sanomaly=sanomaly,
            mean_data=None,
            std_data=None,
            wsz=p["wsz"],
            stride=p["stride"],
        )
        mean, std = dataset.get_info()

    elif p["train_db_name"] == "psm":
        from data.PSM import PSM

        dataset = PSM(
            train=True,
            transform=transform,
            sanomaly=sanomaly,
            mean_data=None,
            std_data=None,
            wsz=p["wsz"],
            stride=p["stride"],
        )
        mean, std = dataset.get_info()
    
    elif p["train_db_name"] == "new_smd":
        from data.new_SMD import SMD

        dataset = SMD(
            p["fname"],
            train=True,
            transform=transform,
            sanomaly=sanomaly,
            mean_data=None,
            std_data=None,
            wsz=p["wsz"],
            stride=p["stride"],
        )
        mean, std = dataset.get_info()
    else:
        raise ValueError("Invalid train dataset {}".format(p["train_db_name"]))

    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset:  # Dataset returns a ts and an augmentation of that.
        if "new" in p["train_db_name"]:
            from data.new_custom_dataset import AugmentedDataset
        else:
            from data.custom_dataset import AugmentedDataset

        dataset = AugmentedDataset(dataset)

    if (
        to_neighbors_dataset
    ):  # Dataset returns ts and its nearest and furthest neighbors.
        from data.custom_dataset import NeighborsDataset

        nindices = np.load(p["topk_neighbors_train_path"])
        findices = np.load(p["bottomk_neighbors_train_path"])
        dataset = NeighborsDataset(dataset, None, nindices, findices, p)

    dataset.mean = mean
    dataset.std = std
    return dataset


def get_aug_train_dataset(p, transform, dataset=None, new=False, data_number=None):
    if new:
        if "new" in p["train_db_name"]:
            from data.new_custom_dataset import ContrustiveDataset, DynamicNeighbors
        else:
            from data.ra_dataset import DynamicNeighbors
            from data.custom_dataset import ContrustiveDataset
        assert dataset is not None
        dynamic_dataset = DynamicNeighbors(dataset, p, data_number=data_number)
        con_dataset = ContrustiveDataset(dynamic_dataset, transform, p)
        return dynamic_dataset, con_dataset
    if dataset is None:
        dataset = torch.load(p["contrastive_dataset"], weights_only=False).dataset
    from data.custom_dataset import NeighborsDataset

    N_indices = np.load(p["topk_neighbors_train_path"])
    F_indices = np.load(p["bottomk_neighbors_train_path"])
    dataset = NeighborsDataset(dataset, transform, N_indices, F_indices, p)

    return dataset


def get_val_dataset(
    p,
    transform=None,
    sanomaly=None,
    to_neighbors_dataset=False,
    mean_data=None,
    std_data=None,
    data=None,
    label=None,
):
    # Base dataset
    if p["val_db_name"] == "smd":
        from data.SMD import SMD

        dataset = SMD(
            p["fname"],
            train=False,
            transform=transform,
            sanomaly=sanomaly,
            mean_data=mean_data,
            std_data=std_data,
            wsz=p["wsz"],
            stride=p["stride"],
        )

    elif p["val_db_name"] == "psm":
        from data.PSM import PSM

        dataset = PSM(
            train=False,
            transform=transform,
            sanomaly=sanomaly,
            mean_data=mean_data,
            std_data=std_data,
            wsz=p["wsz"],
            stride=p["stride"],
        )

    else:
        raise ValueError("Invalid validation dataset {}".format(p["val_db_name"]))

    # Wrap into other dataset (__getitem__ changes)
    if to_neighbors_dataset:  # Dataset returns a ts and one of its nearest neighbors.
        from data.custom_dataset import NeighborsDataset

        N_indices = np.load(p["topk_neighbors_val_path"])
        F_indices = np.load(p["bottomk_neighbors_val_path"])
        dataset = NeighborsDataset(
            dataset, transform, N_indices, F_indices, 5
        )  # Only use 5

    return dataset


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


def inject_sub_anomaly(p):
    return SubAnomaly(p["anomaly_kwargs"]["portion"])


def get_train_transformations(p):
    if p["augmentation_strategy"] == "standard":
        # Standard augmentation strategy
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    **p["augmentation_kwargs"]["random_resized_crop"]
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**p["augmentation_kwargs"]["normalize"]),
            ]
        )

    elif p["augmentation_strategy"] == "ts":
        return transforms.Compose(
            [
                NoiseTransformation(p["transformation_kwargs"]["noise_sigma"]),
                # Crop(p['transformation_kwargs']['crop_size'])
            ]
        )

    else:
        raise ValueError(
            "Invalid augmentation strategy {}".format(p["augmentation_strategy"])
        )


def get_val_transformations(p):
    return transforms.Compose(
        [
            transforms.CenterCrop(p["transformation_kwargs"]["crop_size"]),
            transforms.ToTensor(),
            transforms.Normalize(**p["transformation_kwargs"]["normalize"]),
        ]
    )


def get_val_transformations1(p):
    return transforms.Compose(
        [NoiseTransformation(p["transformation_kwargs"]["noise_sigma"])]
    )


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only:  # Only weights in the cluster head will be updated
        for name, param in model.named_parameters():
            if "cluster_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(params) == 2 * p["num_heads"]

    else:
        params = model.parameters()

    if p["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params, **p["optimizer_kwargs"])

    elif p["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params, **p["optimizer_kwargs"])

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

def adjust_learning_rate(p, optimizer, epoch):
    lr = p["optimizer_kwargs"]["lr"]
    warmup_epochs = p["scheduler_kwargs"].get("lr_warmup_epochs", 0)
    if epoch < warmup_epochs:
        lr = lr * (epoch / warmup_epochs)
    else:
        epoch -= warmup_epochs
        if p["scheduler"] == "cosine":
            eta_min = p["scheduler_kwargs"]["lr_eta_min"]
            lr = (
                eta_min
                + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p["epochs"])) / 2
            )
        elif p["scheduler"] == "cosine_restart":
            eta_min = p["scheduler_kwargs"]["lr_eta_min"]
            cycle_period = p["scheduler_kwargs"]["T_period"]
            cycle_period_mul = p["scheduler_kwargs"].get("T_mul", 1)

            cycle = 0
            if epoch < cycle_period:
                cycle = 0
            else:
                epoch -= cycle_period
                cycle = 1
                cycle_period *= cycle_period_mul
                while True:
                    if epoch > cycle_period:
                        epoch -= cycle_period
                        cycle += 1
                        cycle_period *= cycle_period_mul
                    else:
                        break

            lr = (
                eta_min
                + (lr - eta_min) * (1 + math.cos(math.pi * epoch / cycle_period)) / 2
            )
        elif p["scheduler"] == "step":
            steps = np.sum(epoch > np.array(p["scheduler_kwargs"]["lr_decay_epochs"]))
            if steps > 0:
                lr = lr * (p["scheduler_kwargs"]["lr_decay_rate"] ** steps)
        elif p["scheduler"] == "constant":
            lr = lr
        elif p["scheduler"] == "linear":
            lr = lr * (1 - epoch / p["epochs"])
        else:
            raise ValueError("Invalid learning rate schedule {}".format(p["scheduler"]))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return lr
