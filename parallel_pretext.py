import argparse
import os

import numpy as np
import torch

from carla_pretext import set_seed
from utils.common_config import (
    get_criterion,
    get_model,
    get_optimizer,
    get_scheduler,
    get_train_dataloader,
    get_train_dataset,
    get_train_transformations,
    get_val_dataloader,
    get_val_dataset,
    get_val_transformations,
    inject_sub_anomaly,
)
from utils.config import create_config
from utils.evaluate_utils import GradientMonitor, contrastive_evaluate
from utils.utils import AverageMeter, Logger, ProgressMeter, clean_checkpoint


class PretextWorker:
    """Holds everything that is per-experiment: model, criterion, optimizer,
    scheduler, logger, checkpoints and gradient monitor. The datasets and
    dataloaders are shared between workers."""

    def __init__(self, p, device, train_dataloader, val_dataloader, logger_name):
        set_seed(4)

        self.p = p
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.current_lr = None
        self.avg_meters = None
        self.progress = None
        self.epoch_losses = None

        self.logger = Logger(
            p["version"], verbose=2, file_path=p["pretext_dir"], use_tensorboard=True, delete_files=True, name=logger_name
        )
        self.logger.log("CARLA Pretext stage --> ")
        self.logger.log_hyperparams(p)

        self.model = get_model(p)
        self.logger.add_graph(self.model, torch.rand((1, p['res_kwargs']['in_channels'], p['wsz'])))
        self.model = self.model.to(device)

        self.criterion = get_criterion(p)
        self.criterion = self.criterion.to(device)

        self.optimizer = get_optimizer(p, self.model)
        self.scheduler = get_scheduler(p, self.optimizer)

        self.best_metrics = {
            "loss": np.inf,
            "clear_loss": np.inf,
            "train_calinski": -np.inf,
            "train_davies": np.inf,
            "train_silhouette": -np.inf,
            "eval_calinski": -np.inf,
            "eval_davies": np.inf,
            "eval_silhouette": -np.inf,
        }

        if os.path.exists(p["pretext_checkpoint"]):
            self.logger.log("Restart from checkpoint {}".format(p["pretext_checkpoint"]))
            checkpoint = torch.load(
                p["pretext_checkpoint"], map_location="cpu", weights_only=False
            )
            checkpoint_model = clean_checkpoint(checkpoint["model"], p["pretext_checkpoint"], checkpoint)
            self.model.load_state_dict(checkpoint_model)
            self.model.to(device)
            if "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.start_epoch = checkpoint.get("next_epoch", checkpoint["epoch"] + 1)
            self.best_metrics["loss"] = checkpoint["pretext_best_loss"]
            self.best_metrics["clear_loss"] = checkpoint["pretext_best_clear_loss"] if "pretext_best_clear_loss" in checkpoint else self.best_metrics["clear_loss"]
            self.best_metrics["train_calinski"] = checkpoint["pretext_best_train_calinski"] if "pretext_best_train_calinski" in checkpoint else self.best_metrics["train_calinski"]
            self.best_metrics["train_davies"] = checkpoint["pretext_best_train_davies"] if "pretext_best_train_davies" in checkpoint else self.best_metrics["train_davies"]
            self.best_metrics["train_silhouette"] = checkpoint["pretext_best_train_silhouette"] if "pretext_best_train_silhouette" in checkpoint else self.best_metrics["train_silhouette"]
            self.best_metrics["eval_calinski"] = checkpoint["pretext_best_eval_calinski"] if "pretext_best_eval_calinski" in checkpoint else self.best_metrics["eval_calinski"]
            self.best_metrics["eval_davies"] = checkpoint["pretext_best_eval_davies"] if "pretext_best_eval_davies" in checkpoint else self.best_metrics["eval_davies"]
            self.best_metrics["eval_silhouette"] = checkpoint["pretext_best_eval_silhouette"] if "pretext_best_eval_silhouette" in checkpoint else self.best_metrics["eval_silhouette"]
            self.criterion.update_margin(checkpoint.get("last_margin", None))
            if "prev_ema_loss" in checkpoint:
                self.criterion.prev_ema_loss = checkpoint["prev_ema_loss"]
            if "previous_loss" in checkpoint:
                self.criterion.previous_loss = checkpoint["previous_loss"]

            if self.start_epoch >= p["epochs"] and os.path.exists(p["pretext_model"]):
                checkpoint_model = torch.load(p["pretext_model"], map_location="cpu", weights_only=False)
                checkpoint_model = clean_checkpoint(checkpoint_model, p["pretext_model"])
                self.model.load_state_dict(checkpoint_model)
                self.model.to(device)
                self.start_epoch = p["epochs"]  # skip training if model already exists
            self.gradient_monitor = GradientMonitor(self.model, self.logger, step=self.start_epoch)
        else:
            self.logger.log("No checkpoint file at {}".format(p["pretext_checkpoint"]))
            self.start_epoch = 0
            self.model = self.model.to(device)

            # Make an evaluation run to check random state
            feats, metadata, evaluation_metrics = contrastive_evaluate(
                self.train_dataloader,
                self.model,
                output_metrics=p.get("evaluation_extra_metrics", False),
            )
            self.logger.add_embedding("Cluster", feats, metadata, 0)
            self.logger.metrics_summary("Pretext Evaluation", evaluation_metrics, 0)
            self.gradient_monitor = GradientMonitor(self.model, self.logger, step=0)


def build_shared_datasets(p, dataset_cache):
    """Build train/val datasets once and cache them. Experiments share the
    same cached dataset when they agree on everything that influences the
    data (name, fname, windowing, noise sigma, anomaly portion)."""
    key = (
        p["train_db_name"],
        p["fname"],
        p["wsz"],
        p["stride"],
        p["transformation_kwargs"]["noise_sigma"],
        p["anomaly_kwargs"]["portion"],
    )
    if key in dataset_cache:
        return dataset_cache[key]

    train_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations(p)
    sanomaly = inject_sub_anomaly(p)

    train_dataset = get_train_dataset(
        p, train_transforms, sanomaly, to_augmented_dataset=True
    )
    val_dataset = get_val_dataset(
        p, val_transforms, sanomaly, False, train_dataset.mean, train_dataset.std
    )

    dataset_cache[key] = (train_dataset, val_dataset)
    return dataset_cache[key]


def train_epoch(workers, train_loader, device, epoch):
    """One epoch for all active workers on the same batches.

    Mirrors utils.train_utils.pretext_train, but the batch is moved to the
    device once and every worker is stepped on it. All loss values stay on
    the GPU during the batch loop (losses themselves are sync-free), and
    scalars are only read back for display (every 100 batches) and at the
    end of the epoch, so the GPU pipelines of the different models fully
    overlap."""
    for wk in workers:
        wk.avg_meters = {"meter_Margin": AverageMeter("Margin", ":.4e")}
        wk.progress = ProgressMeter(
            len(train_loader),
            list(wk.avg_meters.values()),
            wk.logger,
            prefix="Epoch: [{}]".format(epoch + 1),
        )
        wk.gpu_sums = {}
        wk.gpu_counts = {}

    def accumulate(wk, key, value):
        if not torch.is_tensor(value):
            value = torch.tensor(float(value), device=device)
        else:
            value = value.detach()
        if key in wk.gpu_sums:
            wk.gpu_sums[key] = wk.gpu_sums[key] + value
            wk.gpu_counts[key] += 1
        else:
            wk.gpu_sums[key] = value
            wk.gpu_counts[key] = 1

    def display_meters(wk):
        for key in wk.gpu_sums:
            meter_name = "meter_Margin" if key == "margin" else f"meter_{key}"
            if meter_name not in wk.avg_meters:
                name = "Margin" if key == "margin" else key
                wk.avg_meters[meter_name] = AverageMeter(name, ":.4e")
                wk.progress.update(wk.avg_meters[meter_name])
            wk.avg_meters[meter_name].update((wk.gpu_sums[key] / wk.gpu_counts[key]).item())

    last_batch_idx = len(train_loader.dataset) // train_loader.batch_size - 1

    for i, batch in enumerate(train_loader):
        ts_org = batch["ts_org"].float().to(device, non_blocking=True)
        ts_w_augmented = batch["ts_w_augment"].float().to(device, non_blocking=True)
        ts_ss_augmented = batch["ts_ss_augment"].float().to(device, non_blocking=True)

        if ts_org.ndim == 3:
            b, w_len, h = ts_org.shape
        else:
            b, w_len = ts_org.shape
            h = 1

        for wk in workers:
            model = wk.model
            wk.optimizer.zero_grad()

            anch_out = model(ts_org.reshape(b, h, w_len))
            nn_out = model(ts_w_augmented.reshape(b, h, w_len))
            model.eval()
            fn_out = model(ts_ss_augmented.view(b, h, w_len))
            model.train()

            output = torch.cat([anch_out, nn_out, fn_out], dim=0)
            losses = wk.criterion(output)

            losses["loss"].backward()
            if (wk.gradient_monitor is not None) and (i == last_batch_idx):
                wk.gradient_monitor.step()
            wk.optimizer.step()

            for key in losses.keys():
                accumulate(wk, key, losses[key])
            accumulate(wk, "margin", wk.criterion.margin)

        if i % 100 == 0:
            for wk in workers:
                display_meters(wk)
                wk.progress.display(i)

    for wk in workers:
        display_meters(wk)
        wk.epoch_losses = {
            key: (wk.gpu_sums[key] / wk.gpu_counts[key]).item() for key in wk.gpu_sums
        }


def make_checkpoint(wk, epoch, best_current_metric, metric, metric_name, assending=False):
    p = wk.p
    checkpoint = (metric > best_current_metric) if assending else (metric < best_current_metric)
    if checkpoint:
        best_current_metric = metric
        wk.best_metrics[metric_name] = best_current_metric
        # Keep the generic model path owned by the loss-best checkpoint.
        if metric_name == "loss":
            torch.save(wk.model.state_dict(), p["pretext_model"])
        save_dict = {
            "model": wk.model.state_dict(),
            "optimizer": wk.optimizer.state_dict(),
            "scheduler": wk.scheduler.state_dict(),
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "pretext_best_loss": wk.best_metrics["loss"],
            "pretext_best_clear_loss": wk.best_metrics["clear_loss"],
            "pretext_best_train_calinski": wk.best_metrics["train_calinski"],
            "pretext_best_train_davies": wk.best_metrics["train_davies"],
            "pretext_best_train_silhouette": wk.best_metrics["train_silhouette"],
            "pretext_best_eval_calinski": wk.best_metrics["eval_calinski"],
            "pretext_best_eval_davies": wk.best_metrics["eval_davies"],
            "pretext_best_eval_silhouette": wk.best_metrics["eval_silhouette"],
            "last_margin": wk.epoch_losses["margin"],
        }
        if hasattr(wk.criterion, "prev_ema_loss"):
            save_dict["prev_ema_loss"] = wk.criterion.prev_ema_loss
        if hasattr(wk.criterion, "previous_loss"):
            save_dict["previous_loss"] = wk.criterion.previous_loss
        torch.save(save_dict, f"{p['pretext_checkpoint'][:-4]}_{metric_name}.pth.tar")
    return best_current_metric


def eval_and_checkpoint(wk, epoch, eval_every_n_epoch):
    p = wk.p
    tmp_loss = wk.epoch_losses["loss"]
    if not (
        (epoch + 1) % eval_every_n_epoch == 0
        or (epoch + 1) == p["epochs"]
        or tmp_loss < wk.best_metrics["loss"]
    ):
        return

    wk.logger.metrics_summary("Pretext Loss", wk.epoch_losses, epoch + 1)
    feats, metadata, evaluation_metrics = contrastive_evaluate(
        wk.train_dataloader,
        wk.model,
        output_metrics=p.get("evaluation_extra_metrics", True),
    )
    wk.logger.add_embedding("Cluster", feats, metadata, epoch + 1)
    wk.logger.metrics_summary("Pretext Evaluation", evaluation_metrics, epoch + 1)

    feats, metadata, evaluation_metrics_eval = contrastive_evaluate(
        wk.val_dataloader,
        wk.model,
        output_metrics=p.get("evaluation_extra_metrics", True),
    )
    wk.logger.add_embedding("Cluster_eval", feats, metadata, epoch + 1)
    wk.logger.metrics_summary("Pretext Evaluation_eval", evaluation_metrics_eval, epoch + 1)

    wk.logger.scalar_summary("", "Learning Rate", wk.current_lr, epoch + 1)

    wk.best_metrics["loss"] = make_checkpoint(wk, epoch, wk.best_metrics["loss"], wk.epoch_losses["loss"], "loss", assending=False)
    wk.best_metrics["clear_loss"] = make_checkpoint(wk, epoch, wk.best_metrics["clear_loss"], wk.epoch_losses["clear_loss"], "clear_loss", assending=False)
    wk.best_metrics["train_calinski"] = make_checkpoint(wk, epoch, wk.best_metrics["train_calinski"], evaluation_metrics["Calinski-Harabasz Score"], "train_calinski", assending=True)
    wk.best_metrics["train_davies"] = make_checkpoint(wk, epoch, wk.best_metrics["train_davies"], evaluation_metrics["Davies-Bouldin Score"], "train_davies", assending=False)
    wk.best_metrics["train_silhouette"] = make_checkpoint(wk, epoch, wk.best_metrics["train_silhouette"], evaluation_metrics["Silhouette Score"], "train_silhouette", assending=True)
    wk.best_metrics["eval_calinski"] = make_checkpoint(wk, epoch, wk.best_metrics["eval_calinski"], evaluation_metrics_eval["Calinski-Harabasz Score"], "eval_calinski", assending=True)
    wk.best_metrics["eval_davies"] = make_checkpoint(wk, epoch, wk.best_metrics["eval_davies"], evaluation_metrics_eval["Davies-Bouldin Score"], "eval_davies", assending=False)
    wk.best_metrics["eval_silhouette"] = make_checkpoint(wk, epoch, wk.best_metrics["eval_silhouette"], evaluation_metrics_eval["Silhouette Score"], "eval_silhouette", assending=True)

    save_dict = {
        "model": wk.model.state_dict(),
        "optimizer": wk.optimizer.state_dict(),
        "scheduler": wk.scheduler.state_dict(),
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "pretext_best_loss": wk.epoch_losses["loss"],
        "pretext_best_clear_loss": wk.epoch_losses["clear_loss"],
        "pretext_best_train_calinski": evaluation_metrics_eval["Calinski-Harabasz Score"],
        "pretext_best_train_davies": evaluation_metrics_eval["Davies-Bouldin Score"],
        "pretext_best_train_silhouette": evaluation_metrics_eval["Silhouette Score"],
        "pretext_best_eval_calinski": evaluation_metrics["Calinski-Harabasz Score"],
        "pretext_best_eval_davies": evaluation_metrics["Davies-Bouldin Score"],
        "pretext_best_eval_silhouette": evaluation_metrics["Silhouette Score"],
        "last_margin": wk.epoch_losses["margin"],
    }
    if hasattr(wk.criterion, "prev_ema_loss"):
        save_dict["prev_ema_loss"] = wk.criterion.prev_ema_loss
    if hasattr(wk.criterion, "previous_loss"):
        save_dict["previous_loss"] = wk.criterion.previous_loss
    torch.save(save_dict, p["pretext_checkpoint"])


def run(workers, train_loader, device, eval_every_n_epoch=200):
    max_epochs = max(wk.p["epochs"] for wk in workers)
    for epoch in range(max_epochs):
        active = [wk for wk in workers if wk.start_epoch <= epoch < wk.p["epochs"]]
        if not active:
            continue

        for wk in active:
            wk.logger.log("Epoch %d/%d" % (epoch + 1, wk.p["epochs"]))
            wk.logger.log("-" * 15)
            wk.current_lr = wk.optimizer.param_groups[0]["lr"]
            wk.logger.log("Adjusted learning rate to {:.5f}".format(wk.current_lr))

        train_epoch(active, train_loader, device, epoch)

        for wk in active:
            wk.scheduler.step()
            eval_and_checkpoint(wk, epoch, eval_every_n_epoch)

    for wk in workers:
        wk.logger.finalize(eval_every_n_epoch)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    configs = []
    for config_exp in args.config_exp:
        exp_name = os.path.basename(config_exp)[: -len(".yml")]
        version = os.path.join(args.version, exp_name) if args.version else exp_name
        p = create_config(args.config_env, config_exp, args.fname, version)
        p["experiment_name"] = exp_name
        configs.append(p)

    ref = configs[0]
    for p in configs[1:]:
        for key in ["train_db_name", "val_db_name", "fname", "wsz", "stride", "batch_size", "num_workers"]:
            if p[key] != ref[key]:
                raise ValueError(
                    "All experiments must share '{}' to train on shared batches "
                    "('{}' vs '{}').".format(key, ref[key], p[key])
                )
        if (
            p["transformation_kwargs"]["noise_sigma"] != ref["transformation_kwargs"]["noise_sigma"]
            or p["anomaly_kwargs"]["portion"] != ref["anomaly_kwargs"]["portion"]
        ):
            raise ValueError(
                "All experiments must share 'transformation_kwargs.noise_sigma' and "
                "'anomaly_kwargs.portion' so the augmented dataset (the RAM-heavy part) "
                "can be shared."
            )

    dataset_cache = {}
    train_dataset, val_dataset = build_shared_datasets(ref, dataset_cache)
    train_dataloader = get_train_dataloader(ref, train_dataset)
    val_dataloader = get_val_dataloader(ref, val_dataset)
    print(
        "Dataset contains {}/{} train/val samples".format(len(train_dataset), len(val_dataset))
    )

    workers = []
    for idx, p in enumerate(configs):
        logger_name = "Self-Awareness-{}-{}-{}".format(idx, p["experiment_name"], p["fname"])
        workers.append(PretextWorker(p, device, train_dataloader, val_dataloader, logger_name))

    run(workers, train_dataloader, device)


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="parallel pretext")
    parser.add_argument("--config_env", help="Config file for the environment")
    parser.add_argument("--config_exp", nargs="+", help="Config files for the experiments")
    parser.add_argument("--fname", help="Config the file name of Dataset")
    parser.add_argument("--version", help="Base experiment version; the config file name is appended per experiment", type=str, default=None)
    parser.add_argument("--device", help="Torch device to train on", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)
