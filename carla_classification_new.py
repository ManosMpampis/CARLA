import argparse
import os
import torch
import numpy as np
from utils.config import create_config
from utils.common_config import (
    get_train_transformations,
    get_val_transformations1,
    get_train_dataset,
    get_train_dataloader,
    get_aug_train_dataset,
    get_val_dataset,
    get_val_dataloader,
    get_optimizer,
    get_scheduler,
    get_model,
    get_criterion,
    adjust_learning_rate,
    inject_sub_anomaly,
)
from utils.evaluate_utils import get_predictions, pr_evaluate
from utils.train_utils import self_sup_classification_train
from utils.utils import Logger, clean_checkpoint
from utils.ts_figures import make_figures

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
    global best_f1
    p = create_config(args.config_env, args.config_exp, args.fname, args.version)
    logger = Logger(
        p["version"], verbose=2, file_path=p["classification_dir"], use_tensorboard=True
    )

    logger.log("CARLA Self-supervised Classification stage --> ")
    logger.log_hyperparams(p)

    # Data
    logger.log(
        "\n- Get dataset and dataloaders for "
        + p["train_db_name"]
        + " dataset - timeseries "
        + p["fname"]
    )
    train_transforms = get_train_transformations(p)
    sanomaly = inject_sub_anomaly(p)
    val_transforms = get_val_transformations1(p)

    train_dataset = get_train_dataset(
        p, train_transforms, sanomaly, to_augmented_dataset=True
    )
    val_dataset = get_val_dataset(
        p, val_transforms, sanomaly, False, train_dataset.mean, train_dataset.std
    )

    # train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)

    # Model
    model = get_model(p, p["pretext_model"])
    model = model.to(device)

    # Find new datase
    train_dataset_base, con_dataset = get_aug_train_dataset(p, transform=train_transforms, dataset=train_dataset, new=True, data_number=None)

    train_dataloader = get_train_dataloader(p, con_dataset)
    base_dataloader = get_val_dataloader(p, train_dataset_base)

    logger.log(
        "-- Train samples size: %d - Test samples size: %d"
        % (len(train_dataset), len(val_dataset))
    )

    # Optimizer
    optimizer = get_optimizer(p, model, p["update_cluster_head_only"])
    scheduler = get_scheduler(p, optimizer)

    # Warning
    if p["update_cluster_head_only"]:
        logger.log("WARNING: classification will only update the cluster head")

    # Loss function
    criterion = get_criterion(p)
    criterion.to(device)

    logger.log("\n- Model initialisation")
    # Initi neighbors with the current model
    predictions = train_dataset_base.predict_and_update(model, base_dataloader, p)

    # Checkpoint
    if os.path.exists(p["classification_checkpoint"]):
        logger.log(
            "-- Model initialised from last checkpoint: {}".format(
                p["classification_checkpoint"]
            )
        )
        checkpoint = torch.load(
            p["classification_checkpoint"], map_location="cpu", weights_only=False
        )
        if "scheduler" in checkpoint.keys():
            scheduler.load_state_dict(checkpoint["scheduler"])
        model_checkpoint = clean_checkpoint(checkpoint["model"], p["classification_checkpoint"], checkpoint)
        model.load_state_dict(model_checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        normal_label = checkpoint["normal_label"]
        best_f1 = checkpoint["best_f1"]
        best_cls_f1 = checkpoint.get("best_cls_f1", -1 * np.inf)
        train_best_f1 = checkpoint.get("train_best_f1", -1 * np.inf)

        if start_epoch >= p["epochs"]-1 and os.path.exists(p["classification_model"]):
            checkpoint = torch.load(p["classification_model"], map_location="cpu", weights_only=False)
            model_checkpoint = clean_checkpoint(checkpoint["model"], p["classification_model"], checkpoint)
            model.load_state_dict(model_checkpoint)
            model.to(device)
            normal_label = checkpoint["normal_label"]
            start_epoch = p["epochs"] + 1  # skip training if model already exists

    else:
        logger.log(
            "-- No checkpoint file at {} -- new model initialised".format(
                p["classification_checkpoint"]
            )
        )
        start_epoch = 0
        normal_label = 0
        best_f1 = -1 * np.inf
        best_cls_f1 = -1 * np.inf
        train_best_f1 = -1 * np.inf
    
    # Initi neighbors with the current model
    # predictions = train_dataset_base.predict_and_update(model, base_dataloader, p)
    logger.log("\n- Training:")
    for epoch in range(start_epoch, p["epochs"]):
        logger.log("-- Epoch %d/%d" % (epoch + 1, p["epochs"]))
        
        lr = optimizer.param_groups[0]["lr"]
        loss_dict = self_sup_classification_train(
            train_dataloader,
            model,
            criterion,
            optimizer,
            epoch,
            logger,
            p["update_cluster_head_only"],
        )

        predictions = train_dataset_base.predict_and_update(model, base_dataloader, p, p["update_data"])

        label_counts = torch.bincount(predictions["predictions"])
        normal_label = 0 if p['setup'] == 'classification_e2e' else label_counts.argmax()

        train_metrics = pr_evaluate(
            predictions, majority_label=normal_label, train=True
        )

        train_rep_f1 = train_metrics["best_f1"]
        predictions = get_predictions(p, val_dataloader, model, False, False)

        eval_metrics = pr_evaluate(
            predictions, majority_label=normal_label
        )
        rep_f1 = eval_metrics["best_f1"]
        scheduler.step()

        if epoch % 100 == 0 or epoch == p["epochs"] - 1 or rep_f1 >= best_f1 or eval_metrics["cls_f1"] >= best_cls_f1:
            print(f"log at epoch: {epoch}/{p["epochs"]}")
            logger.scalar_summary("", "Learning Rate", lr, epoch)
            logger.metrics_summary("Classification Evaluation", eval_metrics, epoch)
            logger.metrics_summary("Classification Train", train_metrics, epoch)
            report_str = (
                f"\nValidation Set Metrics\n"
                f"Anomalies Classification --> TP: {eval_metrics['cls_tp']}, TN: {eval_metrics['cls_tn']}, FN: {eval_metrics['cls_fn']}, FP: {eval_metrics['cls_fp']}\n"
                f"Anomalies Best F1 --> TP: {eval_metrics['best_tp']}, TN: {eval_metrics['best_tn']}, FN: {eval_metrics['best_fn']}, FP: {eval_metrics['best_fp']}\n"
                f"Majority label: {normal_label}"
            )
            logger.log(report_str)
            logger.metrics_summary("Classification Loss", loss_dict, epoch)
            # Function that makes and logs figures to tensorboard
            # Needs to find a way that the inputs have the whole timeseries
            # While labels and predictions correspond to one time interval.
            # make_figures(logger, inputs, labels, predictions, mode="Validation", epoch=epoch)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "normal_label": normal_label,
                    "best_f1": best_f1,
                    "best_cls_f1": best_cls_f1,
                    "train_best_f1": train_best_f1,
                },
                p["classification_checkpoint"],
            )

        if eval_metrics["cls_f1"] >= best_cls_f1:
            best_cls_f1 = eval_metrics["cls_f1"]

        if rep_f1 >= best_f1:
            best_f1 = rep_f1
            # logger.log('New Checkpoint ...')
            torch.save(
                {"model": model.state_dict(), "normal_label": normal_label},
                p["classification_model"],
            )
        
        if train_rep_f1 >= train_best_f1:
            train_best_f1 = train_rep_f1
            # logger.log('New Checkpoint ...')
            torch.save(
                {"model": model.state_dict(), "normal_label": normal_label},
                f"{p["classification_model"][:-8]}_train.pth.tar",
            )

    model_checkpoint = torch.load(
        p["classification_model"], map_location="cpu", weights_only=False
    )
    model.load_state_dict(model_checkpoint["model"])
    normal_label = model_checkpoint["normal_label"]

    predictions, _ = get_predictions(p, val_dataloader, model, True)
    eval_metrics = pr_evaluate(
            predictions, majority_label=normal_label
        )
    
    predictions = train_dataset_base.predict_and_update(model, base_dataloader, p, False)
    logger.finalize()


if __name__ == "__main__":
    FLAGS = argparse.ArgumentParser(description="classification Loss")
    FLAGS.add_argument("--config_env", help="Location of path config file")
    FLAGS.add_argument("--config_exp", help="Location of experiments config file")
    FLAGS.add_argument("--fname", help="Config the file name of Dataset")
    FLAGS.add_argument("--version", help="Experiment version", type=str)
    args = FLAGS.parse_args()
    main(args)
