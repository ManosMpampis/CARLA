import argparse
import os
import csv
import torch
import numpy as np
from utils.config import create_config
from utils.common_config import (
    get_train_transformations,
    get_val_transformations,
    get_train_dataset,
    get_train_dataloader,
    get_aug_train_dataset,
    get_val_dataset,
    get_val_dataloader,
    get_optimizer,
    get_scheduler,
    get_model,
    get_criterion,
    inject_sub_anomaly,
)
from utils.evaluate_utils import get_predictions, pr_evaluate, pr_evaluate_timeseries, GradientMonitor
from utils.train_utils import self_sup_classification_train
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


def main(args, update_dictionary={}):
    p = create_config(args.config_env, args.config_exp, args.fname, args.version, update_dictionary=update_dictionary)
    device = torch.device("cuda:0" if torch.cuda.is_available() and p.get("device", "cuda") else "cpu")
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
    val_transforms = get_val_transformations(p)

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
    logger.add_graph(model, torch.rand([1, p['res_kwargs']['in_channels'], p['wsz']]))

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
        model_checkpoint = clean_checkpoint(checkpoint["model"], p["classification_checkpoint"], checkpoint)
        model.load_state_dict(model_checkpoint)
        model.to(device)
        if "scheduler" in checkpoint.keys():
            scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        normal_label = checkpoint["normal_label"]
        best_f1 = checkpoint["best_f1"]
        best_cls_f1 = checkpoint.get("best_cls_f1", -1 * np.inf)
        train_best_f1 = checkpoint.get("train_best_f1", -1 * np.inf)
        best_train_th_eval_f1 = checkpoint.get("best_train_th_eval_f1", -1 * np.inf)

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
        best_train_th_eval_f1 = -1 * np.inf
    
    gradient_monitor = GradientMonitor(model, logger)
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
            device=device,
            gradient_monitor=gradient_monitor
        )

        predictions = train_dataset_base.predict_and_update(model, base_dataloader, p, p["update_data"])

        label_counts = torch.bincount(predictions["predictions"])
        normal_label = 0 if p['setup'] == 'classification_e2e' else label_counts.argmax()

        cls_train_metrics, best_train_metrics, _, best_train_th = pr_evaluate(
            predictions, majority_label=normal_label, train=True
        )

        train_rep_f1 = best_train_metrics["f1_score"]
        predictions = get_predictions(p, val_dataloader, model, False, False)

        cls_eval_metrics, best_eval_metrics, eval_best_metrics_with_train_th, _ = pr_evaluate(
            predictions, majority_label=normal_label, train_best_threshold=best_train_th
        )
        rep_f1 = best_eval_metrics["f1_score"]
        scheduler.step()

        if epoch % 100 == 0 or epoch == p["epochs"] - 1 or rep_f1 >= best_f1 or cls_eval_metrics["f1_score"] > best_cls_f1 or eval_best_metrics_with_train_th["f1_score"] > best_train_th_eval_f1:
            print(f"log at epoch: {epoch}/{p["epochs"]}")
            logger.scalar_summary("", "Learning Rate", lr, epoch)
            logger.metrics_summary("Evaluation cls_eval_metrics Metrics", cls_eval_metrics, epoch)
            logger.metrics_summary("Evaluation Anomaly Score Metrics", best_eval_metrics, epoch)
            logger.metrics_summary("Evaluation Anomaly Score From Training Metrics", eval_best_metrics_with_train_th, epoch)

            logger.metrics_summary("Train Classification Train", cls_train_metrics, epoch)
            logger.metrics_summary("Train Anomaly Score Metrics", best_train_metrics, epoch)

            report_str = (
                f"\nValidation Set Metrics\n"
                f"Anomalies Classification --> Majority Label: {normal_label}\n"
                f"{''.join(f'{key}:{value}\n' for key, value in cls_eval_metrics.items())}"
                f"Anomalies Best F1 --> Best Threshold: {best_train_th}\n"
                f"{''.join(f'{key}:{value}\n' for key, value in best_eval_metrics.items())}"
                f"Anomalies Train Best F1 --> Threshold: {best_train_th}\n"
                f"{''.join(f'{key}:{value}\n' for key, value in eval_best_metrics_with_train_th.items())}"
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
                    "best_train_th_eval_f1": best_train_th_eval_f1,
                    "best_train_threshold": best_train_th
                },
                p["classification_checkpoint"],
            )

        if cls_eval_metrics["f1_score"] >= best_cls_f1:
            best_cls_f1 = cls_eval_metrics["f1_score"]
            torch.save(
                {"model": model.state_dict(), "normal_label": normal_label},
                f"{p["classification_model"][:-8]}_cls.pth.tar",
            )

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
        
        if eval_best_metrics_with_train_th["f1_score"] >= best_train_th_eval_f1:
            best_train_th_eval_f1 = eval_best_metrics_with_train_th["f1_score"]
            # logger.log('New Checkpoint ...')
            torch.save(
                {"model": model.state_dict(), "normal_label": normal_label},
                f"{p["classification_model"][:-8]}_train.pth.tar",
            )
    
    model_checkpoint = torch.load(
        p["classification_model"], map_location="cpu", weights_only=False
    )
    model.load_state_dict(model_checkpoint["model"])

    model_evaluation(model, p, train_dataset_base, base_dataloader, val_dataloader, logger, tag="")
    model_checkpoint = torch.load(
        f"{p["classification_model"][:-8]}_cls.pth.tar", map_location="cpu", weights_only=False
    )
    model.load_state_dict(model_checkpoint["model"])
    model_evaluation(model, p, train_dataset_base, base_dataloader, val_dataloader, logger, tag="cls")

    logger.finalize()

def model_evaluation(model, p, train_dataset_base, base_dataloader, val_dataloader, logger, tag):

    # Evaluate on the train set and find the best threshold for the train set to evaluate on the test set with the same threshold
    predictions = train_dataset_base.predict_and_update(model, base_dataloader, p, False)
    label_counts = torch.bincount(predictions["predictions"])
    normal_label = 0 if p['setup'] == 'classification_e2e' else label_counts.argmax().item()
    cls_train_metrics, best_train_metrics, _, best_train_th = pr_evaluate(
            predictions, majority_label=normal_label, train=True
    )

    with open(p[f"{tag}eval_train_csl"], "w", newline="") as f:
        w = csv.DictWriter(f, cls_train_metrics.keys())
        w.writeheader()
        w.writerow(cls_train_metrics)
    
    with open(p[f"{tag}eval_train_best"], "w", newline="") as f:
        w = csv.DictWriter(f, best_train_metrics.keys())
        w.writeheader()
        w.writerow(best_train_metrics)
    
    predictions, _ = get_predictions(p, val_dataloader, model, True)
    end = predictions["end_idxs"].numpy()
    start = predictions["start_idxs"].numpy()
    labels = predictions["targets"].numpy()
    test_inputs = predictions["inputs"].numpy()
    ts_len = end[-1]

    # Evaluate on every entry as different input
    cls_eval_metrics, best_eval_metrics, best_train_eval_metrics, _ = pr_evaluate(predictions, majority_label=normal_label, train_best_threshold=best_train_th)
    with open(p[f"{tag}eval_test_cls"], "w", newline="") as f:
        w = csv.DictWriter(f, cls_eval_metrics.keys())
        w.writeheader()
        w.writerow(cls_eval_metrics)
    
    with open(p[f"{tag}eval_test_best"], "w", newline="") as f:
        w = csv.DictWriter(f, best_eval_metrics.keys())
        w.writeheader()
        w.writerow(best_eval_metrics)
    
    with open(p[f"{tag}eval_test_train_th"], "w", newline="") as f:
        w = csv.DictWriter(f, best_train_eval_metrics.keys())
        w.writeheader()
        w.writerow(best_train_eval_metrics)
    
    # System evaluation: create a timeseries of predictions and labels and evaluate with the same metrics as the other methods
    gt = np.zeros((ts_len, 1)).astype(int)
    inputs = np.zeros((ts_len, test_inputs[0].shape[-1]))
    for s, e, l, i in zip(start, end, labels, test_inputs):
        gt[s:e] = l.reshape(-1,1)
        inputs[s:e] = i

    cls_score, score_best, score_train_best, threshold_best, best_detections_thresholds = pr_evaluate_timeseries(
        logger,
        predictions,
        best_train_th,
        normal_label,
        gt,
        inputs,
        tag=f"{tag} Final Timeseries_",
        epoch=-2
    )
    
    with open(p[f"{tag}eval_tstest_cls"], "w", newline="") as f:
        w = csv.DictWriter(f, cls_score.keys())
        w.writeheader()
        w.writerow(cls_score)

    with open(p[f"{tag}eval_tstest_best"], "w", newline="") as f:
        w = csv.DictWriter(f, score_best.keys())
        w.writeheader()
        w.writerow(score_best)
    
    with open(p[f"{tag}eval_tstest_trainth"], "w", newline="") as f:
        w = csv.DictWriter(f, score_train_best.keys())
        w.writeheader()
        w.writerow(score_train_best)
    
    report_str = (
                f"\nValidation Set Metrics {tag}\n"
                f"Anomalies Classification --> Majority Label: {normal_label}/Best Detections Threshold:{best_detections_thresholds}\n"
                f"{''.join(f'{key}:{value}\n' for key, value in cls_score.items())}"
                f"Anomalies Best F1 --> Best Threshold: {threshold_best}\n"
                f"{''.join(f'{key}:{value}\n' for key, value in score_best.items())}"
                f"Anomalies Train Best F1 --> Threshold: {best_train_th}\n"
                f"{''.join(f'{key}:{value}\n' for key, value in score_train_best.items())}"
            )
    logger.log(report_str)


if __name__ == "__main__":
    FLAGS = argparse.ArgumentParser(description="classification Loss")
    FLAGS.add_argument("--config_env", help="Location of path config file")
    FLAGS.add_argument("--config_exp", help="Location of experiments config file")
    FLAGS.add_argument("--fname", help="Config the file name of Dataset")
    FLAGS.add_argument("--version", help="Experiment version", type=str)
    args = FLAGS.parse_args()
    main(args)
