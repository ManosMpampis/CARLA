import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import (
    get_criterion,
    get_model,
    get_train_dataset,
    get_val_dataset,
    get_train_dataloader,
    get_val_dataloader,
    get_train_transformations,
    get_val_transformations1,
    adjust_learning_rate,
    inject_sub_anomaly,
    get_scheduler,
)
from utils.evaluate_utils import contrastive_evaluate
from utils.repository import TSRepository, fill_ts_repository
from utils.train_utils import pretext_train
from utils.utils import Logger, clean_checkpoint

import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(4)

device = torch.device("cuda")


def main(args):
    p = create_config(args.config_env, args.config_exp, args.fname, args.version)
    logger = Logger(
        p["version"], verbose=2, file_path=p["pretext_dir"], use_tensorboard=True
    )
    logger.log("CARLA Pretext stage --> ")

    logger.log_hyperparams(p)

    model = get_model(p)
    # logger.add_graph(model, torch.rand(p['res_kwargs']['in_channels'], p['wsz']).unsqueeze(0))
    model = model.to(device)

    train_transforms = get_train_transformations(p)

    sanomaly = inject_sub_anomaly(p)
    val_transforms = get_val_transformations1(p)

    train_dataset = get_train_dataset(
        p, train_transforms, sanomaly, to_augmented_dataset=True
    )
    len_train = len(train_dataset)

    val_dataset = get_val_dataset(
        p, val_transforms, sanomaly, False, train_dataset.mean, train_dataset.std
    )

    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    base_dataloader = get_val_dataloader(p, train_dataset)

    logger.log(
        "Dataset contains {}/{} train/val samples".format(
            len(train_dataset), len(val_dataset)
        )
    )

    ts_repository_base = TSRepository(len_train, p["model_kwargs"], p["res_kwargs"])
    ts_repository_base.to(device)
    ts_repository_val = TSRepository(
        len(val_dataset), p["model_kwargs"], p["res_kwargs"]
    )
    ts_repository_val.to(device)

    criterion = get_criterion(p)
    criterion = criterion.to(device)

    # optimizer = get_optimizer(p, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["optimizer_kwargs"]["lr"])
    scheduler = get_scheduler(p, optimizer)

    # Checkpoint
    if os.path.exists(p["pretext_checkpoint"]):
        logger.log("Restart from checkpoint {}".format(p["pretext_checkpoint"]))
        checkpoint = torch.load(
            p["pretext_checkpoint"], map_location="cpu", weights_only=False
        )
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        checkpoint_model = clean_checkpoint(checkpoint["model"], p["pretext_checkpoint"], checkpoint)
        model.load_state_dict(checkpoint_model)
        model.to(device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        pretext_best_loss = checkpoint["pretext_best_loss"]
        criterion.update_margin(checkpoint.get("last_margin", None))
        if "prev_ema_loss" in checkpoint:
            criterion.prev_ema_loss = checkpoint["prev_ema_loss"]
        if "previous_loss" in checkpoint:
            criterion.previous_loss = checkpoint["previous_loss"]
        
        if start_epoch >= p["epochs"]-1 and os.path.exists(p["pretext_model"]):
            checkpoint_model = torch.load(p["pretext_model"], map_location="cpu", weights_only=False)
            checkpoint_model = clean_checkpoint(checkpoint_model, p["pretext_model"])
            model.load_state_dict(checkpoint_model)
            model.to(device)
            start_epoch = p["epochs"] + 1  # skip training if model already exists

    else:
        logger.log("No checkpoint file at {}".format(p["pretext_checkpoint"]))
        start_epoch = 0
        model = model.to(device)
        pretext_best_loss = np.inf

        # Make an evaluation run to check random state
        feats, metadata, evaluation_metrics = contrastive_evaluate(
            train_dataloader,
            model,
            output_metrics=p.get("evaluation_extra_metrics", False),
        )
        logger.add_embedding("Cluster", feats, metadata, 0)
        logger.metrics_summary("Pretext Evaluation", evaluation_metrics, 0)

    # Training
    pretext_best_loss = np.inf
    for epoch in range(start_epoch, p["epochs"]):
        logger.log("Epoch %d/%d" % (epoch + 1, p["epochs"]))
        logger.log("-" * 15)

        # lr = adjust_learning_rate(p, optimizer, epoch)
        lr = optimizer.param_groups[0]["lr"]
        logger.log("Adjusted learning rate to {:.5f}".format(lr))

        # logger.log('EPOCH ----> ', epoch)
        loss_dict = pretext_train(
            train_dataloader, model, criterion, optimizer, epoch, logger, device=device
        )
        last_margin = loss_dict["margin"]
        tmp_loss = loss_dict["loss"]

        if (
            epoch % 200 == 0
            or epoch == p["epochs"] - 1
            or tmp_loss <= pretext_best_loss
        ):
            logger.metrics_summary("Pretext Loss", loss_dict, epoch + 1)
            feats, metadata, evaluation_metrics = contrastive_evaluate(
                train_dataloader,
                model,
                output_metrics=p.get("evaluation_extra_metrics", False),
            )
            logger.add_embedding("Cluster", feats, metadata, epoch + 1)
            logger.metrics_summary("Pretext Evaluation", evaluation_metrics, epoch + 1)
            logger.scalar_summary("", "Learning Rate", lr, epoch + 1)

            pretext_best_loss = tmp_loss
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "pretext_best_loss": pretext_best_loss,
                "last_margin": last_margin,
            }
            if hasattr(criterion, "prev_ema_loss"):
                save_dict["prev_ema_loss"] = criterion.prev_ema_loss
            if hasattr(criterion, "previous_loss"):
                save_dict["previous_loss"] = criterion.previous_loss
            torch.save(save_dict, p["pretext_checkpoint"])

        # Checkpoint
        if tmp_loss <= pretext_best_loss:
            pretext_best_loss = tmp_loss
            torch.save(model.state_dict(), p["pretext_model"])
        scheduler.step()

    # load final model
    checkpoint = torch.load(
        p["pretext_model"], map_location="cpu", weights_only=False
    )    
    model.load_state_dict(checkpoint)
    model.to(device)

    # Relesase some memory
    del train_dataloader
    # Mine the topk nearest neighbors at the very end (Train)
    # These will be served as input to the classification loss.
    logger.log(
        "Fill TS Repository for mining the nearest/furthest neighbors (train) ..."
    )
    ts_repository_aug = TSRepository(
        len_train * 2, p["model_kwargs"], p["res_kwargs"]
    )  # need size of repository == 1+num_of_anomalies
    fill_ts_repository(
        p,
        base_dataloader,
        model,
        ts_repository_base,
        real_aug=True,
        ts_repository_aug=ts_repository_aug,
    )
    del base_dataloader, train_dataset
    # out_pre = np.column_stack((ts_repository_base.features, ts_repository_base.targets))
    out_pre = np.column_stack(
        (
            ts_repository_base.features.cpu().numpy(),
            ts_repository_base.targets.cpu().numpy(),
        )
    )

    np.save(p["pretext_features_train_path"], out_pre)
    topk = 10
    logger.log("Mine the nearest neighbors (Top-%d)" % (topk))
    kfurtherst, knearest = ts_repository_aug.furthest_nearest_neighbors(topk)
    np.save(p["topk_neighbors_train_path"], knearest)
    np.save(p["bottomk_neighbors_train_path"], kfurtherst)
    del ts_repository_aug, kfurtherst, knearest

    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    logger.log("Fill TS Repository for mining the nearest/furthest neighbors (val) ...")

    fill_ts_repository(
        p,
        val_dataloader,
        model,
        ts_repository_val,
        real_aug=False,
        ts_repository_aug=None,
    )
    # out_pre = np.column_stack((ts_repository_val.features, ts_repository_val.targets))
    del val_dataloader, val_dataset, model
    out_pre = np.column_stack(
        (
            ts_repository_val.features.cpu().numpy(),
            ts_repository_val.targets.cpu().numpy(),
        )
    )

    np.save(p["pretext_features_test_path"], out_pre)
    topk = 10
    logger.log("Mine the nearest and furthest neighbors (Top-%d)" % (topk))
    kfurtherst, knearest = ts_repository_val.furthest_nearest_neighbors(topk)

    np.save(p["topk_neighbors_val_path"], knearest)
    np.save(p["bottomk_neighbors_val_path"], kfurtherst)
    logger.finalize()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="pretext")
    parser.add_argument("--config_env", help="Config file for the environment")
    parser.add_argument("--config_exp", help="Config file for the experiment")
    parser.add_argument("--fname", help="Config the file name of Dataset")
    parser.add_argument("--version", help="Experiment version", type=str)
    args = parser.parse_args()
    main(args)
