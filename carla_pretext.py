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
    get_val_transformations,
    inject_sub_anomaly,
    get_scheduler,
    get_optimizer,
)
from utils.evaluate_utils import contrastive_evaluate, GradientMonitor
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


def main(args, update_dictionary={}):
    p = create_config(args.config_env, args.config_exp, args.fname, args.version, update_dictionary=update_dictionary)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and p.get("device", "cuda")) else "cpu")
    logger = Logger(
        p["version"], verbose=2, file_path=p["pretext_dir"], use_tensorboard=True, delete_files=True
    )
    logger.log("CARLA Pretext stage --> ")

    logger.log_hyperparams(p)

    model = get_model(p)
    logger.add_graph(model, torch.rand((1, p['res_kwargs']['in_channels'], p['wsz'])))
    model = model.to(device)

    train_transforms = get_train_transformations(p)

    sanomaly = inject_sub_anomaly(p)
    val_transforms = get_val_transformations(p)

    train_dataset = get_train_dataset(
        p, train_transforms, sanomaly, to_augmented_dataset=True
    )

    val_dataset = get_val_dataset(
        p, val_transforms, sanomaly, False, train_dataset.mean, train_dataset.std
    )

    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)

    logger.log(
        "Dataset contains {}/{} train/val samples".format(
            len(train_dataset), len(val_dataset)
        )
    )

    criterion = get_criterion(p)
    criterion = criterion.to(device)

    optimizer = get_optimizer(p, model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=p["optimizer_kwargs"]["lr"])
    scheduler = get_scheduler(p, optimizer)

    best_metrics = {
            "loss": np.inf,
            "clear_loss": np.inf,
            "train_calinski": -np.inf,
            "train_davies": np.inf,
            "train_silhouette": -np.inf,
            "eval_calinski": -np.inf,
            "eval_davies": np.inf,
            "eval_silhouette": -np.inf,
            }
    
    # Checkpoint
    if os.path.exists(p["pretext_checkpoint"]):

        logger.log("Restart from checkpoint {}".format(p["pretext_checkpoint"]))
        checkpoint = torch.load(
            p["pretext_checkpoint"], map_location="cpu", weights_only=False
        )
        checkpoint_model = clean_checkpoint(checkpoint["model"], p["pretext_checkpoint"], checkpoint)
        model.load_state_dict(checkpoint_model)
        model.to(device)
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Resume at the next epoch; fall back to the old zero-based format.
        start_epoch = checkpoint.get("next_epoch", checkpoint["epoch"] + 1)
        best_metrics["loss"] = checkpoint["pretext_best_loss"]
        best_metrics["clear_loss"] = checkpoint["pretext_best_clear_loss"] if "pretext_best_clear_loss" in checkpoint else best_metrics["clear_loss"]
        best_metrics["train_calinski"] = checkpoint["pretext_best_train_calinski"] if "pretext_best_train_calinski" in checkpoint else best_metrics["train_calinski"]
        best_metrics["train_davies"] = checkpoint["pretext_best_train_davies"] if "pretext_best_train_davies" in checkpoint else best_metrics["train_davies"]
        best_metrics["train_silhouette"] = checkpoint["pretext_best_train_silhouette"] if "pretext_best_train_silhouette" in checkpoint else best_metrics["train_silhouette"]
        best_metrics["eval_calinski"] = checkpoint["pretext_best_eval_calinski"] if "pretext_best_eval_calinski" in checkpoint else best_metrics["eval_calinski"]
        best_metrics["eval_davies"] = checkpoint["pretext_best_eval_davies"] if "pretext_best_eval_davies" in checkpoint else best_metrics["eval_davies"]
        best_metrics["eval_silhouette"] = checkpoint["pretext_best_eval_silhouette"] if "pretext_best_eval_silhouette" in checkpoint else best_metrics["eval_silhouette"]
        criterion.update_margin(checkpoint.get("last_margin", None))
        if "prev_ema_loss" in checkpoint:
            criterion.prev_ema_loss = checkpoint["prev_ema_loss"]
        if "previous_loss" in checkpoint:
            criterion.previous_loss = checkpoint["previous_loss"]
        
        if start_epoch >= p["epochs"] and os.path.exists(p["pretext_model"]):
            checkpoint_model = torch.load(p["pretext_model"], map_location="cpu", weights_only=False)
            checkpoint_model = clean_checkpoint(checkpoint_model, p["pretext_model"])
            model.load_state_dict(checkpoint_model)
            model.to(device)
            start_epoch = p["epochs"]  # skip training if model already exists
        gradient_monitor = GradientMonitor(model, logger, step=start_epoch)
    else:
        logger.log("No checkpoint file at {}".format(p["pretext_checkpoint"]))
        start_epoch = 0
        model = model.to(device)

        # Make an evaluation run to check random state
        feats, metadata, evaluation_metrics = contrastive_evaluate(
            train_dataloader,
            model,
            output_metrics=p.get("evaluation_extra_metrics", False),
        )
        logger.add_embedding("Cluster", feats, metadata, 0)
        logger.metrics_summary("Pretext Evaluation", evaluation_metrics, 0)
        gradient_monitor = GradientMonitor(model, logger, step=0)

    # Training
    eval_every_n_epoch = 200
    for epoch in range(start_epoch, p["epochs"]):
        logger.log("Epoch %d/%d" % (epoch + 1, p["epochs"]))
        logger.log("-" * 15)

        # lr = adjust_learning_rate(p, optimizer, epoch)
        lr = optimizer.param_groups[0]["lr"]
        logger.log("Adjusted learning rate to {:.5f}".format(lr))

        # logger.log('EPOCH ----> ', epoch)
        loss_dict = pretext_train(
            train_dataloader, model, criterion, optimizer, epoch, logger, device=device, gradient_monitor=gradient_monitor
        )
        last_margin = loss_dict["margin"]
        tmp_loss = loss_dict["loss"]

        # Save the scheduler state for the next epoch in checkpoints.
        scheduler.step()

        if (
            (epoch+1) % eval_every_n_epoch == 0
            or (epoch+1) == p["epochs"]
            or tmp_loss < best_metrics["loss"] #TODO: check edge cases, when model fail to only loss=0, the (<=) saves every epoch. Do we want the last non zero loss?
        ):
            logger.metrics_summary("Pretext Loss", loss_dict, epoch + 1)
            feats, metadata, evaluation_metrics = contrastive_evaluate(
                train_dataloader,
                model,
                output_metrics=p.get("evaluation_extra_metrics", True),
            )
            
            logger.add_embedding("Cluster", feats, metadata, epoch + 1)
            logger.metrics_summary("Pretext Evaluation", evaluation_metrics, epoch + 1)

            feats, metadata, evaluation_metrics_eval = contrastive_evaluate(
                val_dataloader,
                model,
                output_metrics=p.get("evaluation_extra_metrics", True),
            )
            
            logger.add_embedding("Cluster_eval", feats, metadata, epoch + 1)
            logger.metrics_summary("Pretext Evaluation_eval", evaluation_metrics_eval, epoch + 1)

            logger.scalar_summary("", "Learning Rate", lr, epoch + 1)

            # Checkpoint
            best_metrics["loss"] = make_checkpoint(p, epoch, model, optimizer, scheduler, last_margin, criterion, best_metrics, best_metrics["loss"], loss_dict["loss"], "loss", assending=False)

            best_metrics["clear_loss"] = make_checkpoint(p, epoch, model, optimizer, scheduler, last_margin, criterion, best_metrics, best_metrics["clear_loss"], loss_dict["clear_loss"], "clear_loss", assending=False)

            if len(evaluation_metrics)>0:
                best_metrics["train_calinski"] = make_checkpoint(p, epoch, model, optimizer, scheduler, last_margin, criterion, best_metrics, best_metrics["train_calinski"], evaluation_metrics["Calinski-Harabasz Score"], "train_calinski", assending=True)
                best_metrics["train_davies"] = make_checkpoint(p, epoch, model, optimizer, scheduler, last_margin, criterion, best_metrics, best_metrics["train_davies"], evaluation_metrics["Davies-Bouldin Score"], "train_davies", assending=False)
                best_metrics["train_silhouette"] = make_checkpoint(p, epoch, model, optimizer, scheduler, last_margin, criterion, best_metrics, best_metrics["train_silhouette"], evaluation_metrics["Silhouette Score"], "train_silhouette", assending=True)

            if len(evaluation_metrics_eval)>0:
                best_metrics["eval_calinski"] = make_checkpoint(p, epoch, model, optimizer, scheduler, last_margin, criterion, best_metrics, best_metrics["eval_calinski"], evaluation_metrics_eval["Calinski-Harabasz Score"], "eval_calinski", assending=True)
                best_metrics["eval_davies"] = make_checkpoint(p, epoch, model, optimizer, scheduler, last_margin, criterion, best_metrics, best_metrics["eval_davies"], evaluation_metrics_eval["Davies-Bouldin Score"], "eval_davies", assending=False)
                best_metrics["eval_silhouette"] = make_checkpoint(p, epoch, model, optimizer, scheduler, last_margin, criterion, best_metrics, best_metrics["eval_silhouette"], evaluation_metrics_eval["Silhouette Score"], "eval_silhouette", assending=True)
            
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "pretext_best_loss": loss_dict["loss"],
                "pretext_best_clear_loss": loss_dict["clear_loss"],
                "pretext_best_train_calinski": evaluation_metrics_eval.get("Calinski-Harabasz Score", -np.inf),
                "pretext_best_train_davies": evaluation_metrics_eval.get("Davies-Bouldin Score", np.inf),
                "pretext_best_train_silhouette": evaluation_metrics_eval.get("Silhouette Score", -np.inf),
                "pretext_best_eval_calinski": evaluation_metrics.get("Calinski-Harabasz Score", -np.inf),
                "pretext_best_eval_davies": evaluation_metrics.get("Davies-Bouldin Score", np.inf),
                "pretext_best_eval_silhouette": evaluation_metrics.get("Silhouette Score", -np.inf),
                "last_margin": last_margin,
            }
            if hasattr(criterion, "prev_ema_loss"):
                save_dict["prev_ema_loss"] = criterion.prev_ema_loss
            if hasattr(criterion, "previous_loss"):
                save_dict["previous_loss"] = criterion.previous_loss
            torch.save(save_dict, p["pretext_checkpoint"])

            
            # if tmp_loss < pretext_best_loss:
            #     pretext_best_loss = tmp_loss
            #     torch.save(model.state_dict(), p["pretext_model"])
            #     save_dict = {
            #         "model": model.state_dict(),
            #         "optimizer": optimizer.state_dict(),
            #         "scheduler": scheduler.state_dict(),
            #         "epoch": epoch,
            #         "pretext_best_loss": tmp_loss,
            #         "last_margin": last_margin,
            #     }
            #     if hasattr(criterion, "prev_ema_loss"):
            #         save_dict["prev_ema_loss"] = criterion.prev_ema_loss
            #     if hasattr(criterion, "previous_loss"):
            #         save_dict["previous_loss"] = criterion.previous_loss
            #     torch.save(save_dict, f"{p["pretext_checkpoint"][:-4]}_best.pth.tar")
    logger.finalize(eval_every_n_epoch)

def make_checkpoint(p, epoch, model, optimizer, scheduler, last_margin, criterion, best_metrics, best_current_metric, metric, metric_name, assending=False):
    checkpoint = (metric > best_current_metric) if assending else (metric < best_current_metric)
    if checkpoint:
            best_current_metric = metric
            best_metrics[metric_name] = best_current_metric
            # Keep the generic model path owned by the loss-best checkpoint.
            if metric_name == "loss":
                torch.save(model.state_dict(), p["pretext_model"])
            save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "pretext_best_loss": best_metrics["loss"],
                "pretext_best_clear_loss": best_metrics["clear_loss"],
                "pretext_best_train_calinski": best_metrics["train_calinski"],
                "pretext_best_train_davies": best_metrics["train_davies"],
                "pretext_best_train_silhouette": best_metrics["train_silhouette"],
                "pretext_best_eval_calinski": best_metrics["eval_calinski"],
                "pretext_best_eval_davies": best_metrics["eval_davies"],
                "pretext_best_eval_silhouette": best_metrics["eval_silhouette"],
                "last_margin": last_margin,
            }
            if hasattr(criterion, "prev_ema_loss"):
                save_dict["prev_ema_loss"] = criterion.prev_ema_loss
            if hasattr(criterion, "previous_loss"):
                save_dict["previous_loss"] = criterion.previous_loss
            torch.save(save_dict, f"{p["pretext_checkpoint"][:-4]}_{metric_name}.pth.tar")
    return best_current_metric

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="pretext")
    parser.add_argument("--config_env", help="Config file for the environment")
    parser.add_argument("--config_exp", help="Config file for the experiment")
    parser.add_argument("--fname", help="Config the file name of Dataset")
    parser.add_argument("--version", help="Experiment version", type=str)
    args = parser.parse_args()
    main(args)
