import argparse
import json
import os
import random

import numpy as np
import torch

from utils.config import create_config
from utils.common_config import (
    get_criterion,
    get_jepa_model,
    get_jepa_datasets,
    get_optimizer,
    get_scheduler,
    get_train_dataloader,
    get_val_dataloader,
)
from utils.masking import MaskingCollator
from utils.scoring import Scorer, Calibrator
from utils.trainer import Trainer
from utils.utils import Logger


def set_seed(seed: int) -> None:
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


def _build_run(p, device):
    model = get_jepa_model(p)
    criterion = get_criterion(p).to(device)
    optimizer = get_optimizer(p, model)
    scheduler = get_scheduler(p, optimizer)
    return model, criterion, optimizer, scheduler


def _make_logger(p):
    destructive = str(p.get("stage", "pretrain")).lower() != "score"
    logger = Logger(p["version"], verbose=2, file_path=p["jepa_dir"],
                    use_tensorboard=True, delete_files=destructive)
    logger.log(f"CARLA JEPA stage '{p.get('stage', 'pretrain')}' --> ")
    logger.log_hyperparams(p)
    return logger


def _masking_collator(p):
    masking = p.get("stage_a", {}).get("masking", {})
    if masking.get("mode", "none") != "block":
        return None
    kwargs = {k: v for k, v in masking.items() if k != "mode"}
    return MaskingCollator(**kwargs)


class _GraphWrapper(torch.nn.Module):
    """Flattens JEPAModel's nested dict output into a tracer-compatible
    flat {str: Tensor} mapping for TensorBoard graph logging only."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        flat = {}
        for name, z in out["latents"].items():
            flat[f"latent/{name}"] = z
            flat[f"predicted/{name}"] = out["predicted"][name]
            flat[f"target/{name}"] = out["targets"][name]
        return flat


def run_pretrain(p, device):
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
    logger.log(f"Pretraining finished; best val loss {best_val_loss:.6f}")
    logger.finalize()


def run_adapt(p, device):
    logger = _make_logger(p)
    model, criterion, optimizer, scheduler = _build_run(p, device)

    source = p.get("pretrained_from")
    if not source or not os.path.exists(source):
        raise FileNotFoundError(
            f"stage_b adaptation requires 'pretrained_from' checkpoint; got {source}"
        )
    Trainer.load_weights(source, model, logger, strict=True)

    mode = p.get("stage_b", {}).get("mode", "frozen")
    if mode == "frozen":
        for param in model.encoder.parameters():
            param.requires_grad = False
        if model.target_encoder is not None:
            for param in model.target_encoder.parameters():
                param.requires_grad = False
        model.encoder_frozen = True
        frozen = sum(1 for q in model.parameters() if not q.requires_grad)
        logger.log(f"Adaptation mode 'frozen': encoder frozen ({frozen} tensors)")
    elif mode == "finetune":
        logger.log("Adaptation mode 'finetune': all parameters update")
    else:
        raise ValueError(f"Invalid stage_b.mode {mode}")

    model = model.to(device)
    # Rebuild the optimizer AFTER freezing so the param groups match.
    optimizer = get_optimizer(p, model)
    scheduler = get_scheduler(p, optimizer)

    train_dataset, val_dataset = get_jepa_datasets(p)
    train_loader = get_train_dataloader(p, train_dataset)
    val_loader = get_val_dataloader(p, val_dataset)
    logger.log(f"Dataset contains {len(train_dataset)}/{len(val_dataset)} "
               f"train/val samples")

    trainer = Trainer(p, model, criterion, optimizer, scheduler, device, logger,
                      collator=None, amp=p.get("amp", False))
    start_epoch, best_val_loss = Trainer.resume(p, model, optimizer, scheduler, logger)
    best_val_loss = trainer.fit(train_loader, val_loader, start_epoch, best_val_loss)
    logger.log(f"Adaptation finished; best val loss {best_val_loss:.6f}")
    logger.finalize()


def _series_from_dataset(dataset):
    return np.asarray(dataset.series, dtype=np.float32)


@torch.no_grad()
def run_score(p, device):
    from data.jepa_dataset import JEPADataset
    from metrics.metrics import combine_all_evaluation_scores
    from sklearn.metrics import roc_auc_score, average_precision_score

    logger = _make_logger(p)
    model = get_jepa_model(p).to(device)
    weights_path = p.get("score_checkpoint") or p["jepa_model"]
    if not os.path.exists(weights_path):
        weights_path = p["jepa_checkpoint"]
    Trainer.load_weights(weights_path, model, logger, strict=True)
    model.eval()

    scorer = Scorer(model, device)
    calibrator = Calibrator(**p.get("calibration_kwargs",
                                    {"quantile": 0.995}))

    # --- train-side scoring (clean + injected probes): the ONLY calibration input
    train_dataset, _ = get_jepa_datasets(p)
    clean_series = _series_from_dataset(train_dataset)
    clean_result = scorer.score_series(clean_series, p["wsz"], p["stride"])
    clean_channels = {"fused": clean_result["scores"], **clean_result["channels"]}

    probe_channels = None
    probe_cfg = p.get("probe_kwargs", {})
    if probe_cfg.get("num_probe_windows", 0) > 0:
        from data.augment import SubAnomaly

        sanomaly = SubAnomaly(probe_cfg.get("portion", 0.99))
        n_probes = int(probe_cfg["num_probe_windows"])
        rng = np.random.default_rng(p.get("seed", 4))
        idxs = rng.integers(0, len(train_dataset), size=n_probes)
        windows = np.stack([
            sanomaly(train_dataset[int(i)]["ts"]).astype(np.float32)
            for i in idxs
        ])
        batch = torch.from_numpy(windows).permute(0, 2, 1).contiguous()
        probe_scores = scorer.score_windows(batch)
        probe_channels = {
            "fused": probe_scores.pop("fused").reshape(-1),
            **{k: v.reshape(-1) for k, v in probe_scores.pop("levels").items()},
            **{f"signal/{k}": v.reshape(-1) for k, v in probe_scores.pop("signals").items()},
        }

    calibrator.fit(clean_channels, probes=probe_channels)

    fused_clean = calibrator.fuse(clean_channels)
    threshold = calibrator.threshold_for(fused_clean)
    calibrator.save(p["calibration_path"], extra={
        "threshold_fused": threshold,
        "inputs": "clean-train scores only (+ injected-anomaly probes for weights)",
    })
    logger.log(f"Calibration saved to {p['calibration_path']} "
               f"(threshold {threshold:.6g}, fallback={calibrator.fallback})")

    # --- test-side scoring through the same path
    test_dataset = JEPADataset(p, train=False)
    test_series = _series_from_dataset(test_dataset)
    targets = np.asarray(test_dataset.targets).astype(int)
    test_result = scorer.score_series(test_series, p["wsz"], p["stride"])
    test_channels = {"fused": test_result["scores"], **test_result["channels"]}
    fused_test = calibrator.fuse(test_channels)

    pred_labels = (fused_test >= threshold).astype(int)
    window_size = int(p.get("eval_window_size", 100))
    metric_dict = combine_all_evaluation_scores(pred_labels, targets, window_size)

    honest = {
        "window_AUROC": float(roc_auc_score(targets, fused_test)),
        "window_AP": float(average_precision_score(targets, fused_test)),
        "point_precision": float(metric_dict["precision"]),
        "point_recall": float(metric_dict["recall"]),
        "point_F1_no_PA": float(metric_dict["f1_score"]),
        "MCC": float(metric_dict["MCC"]),
    }
    point_adjust = {
        key[3:]: float(value) for key, value in metric_dict.items()
        if key.startswith("pa_") and isinstance(value, (int, float))
    }
    report = {"honest": honest, "point_adjust_comparability": point_adjust}

    np.savez_compressed(
        p["scores_path"],
        scores=fused_test,
        start_idxs=test_result["start_idxs"],
        end_idxs=test_result["end_idxs"],
        cover_counts=test_result["cover_counts"],
        pred_labels=pred_labels,
        gt_labels=targets,
        **{f"channel/{k}": v for k, v in test_channels.items()},
    )
    with open(p["metrics_path"], "w") as f:
        json.dump(report, f, indent=2)

    step = 1
    for section, values in report.items():
        for name, value in values.items():
            logger.scalar_summary(section, name, value, step)
    logger.metrics_summary("Full metric dictionary", {
        k: float(v) for k, v in metric_dict.items()
        if isinstance(v, (int, float))
    }, step)
    logger.log(f"Metrics: honest={honest}")
    logger.log(f"Scores written to {p['scores_path']}, report to {p['metrics_path']}")
    logger.finalize()


STAGES = {
    "pretrain": run_pretrain,
    "pretext": run_pretrain,
    "adapt": run_adapt,
    "score": run_score,
}


def main(args, update_dictionary={}):
    p = create_config(args.config_env, args.config_exp, args.fname, args.version,
                      update_dictionary=update_dictionary)
    set_seed(int(p.get("seed", 4)))
    stage = str(p.get("stage", "pretrain")).lower()
    if stage not in STAGES:
        raise ValueError(f"Invalid stage {stage}; expected one of {sorted(STAGES)}")
    STAGES[stage](p, _device(p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA TSAD harness")
    parser.add_argument("--config_env", help="Config file for the environment")
    parser.add_argument("--config_exp", help="Config file for the experiment")
    parser.add_argument("--fname", help="File name of the dataset machine", default="")
    parser.add_argument("--version", help="Experiment version", type=str)
    args = parser.parse_args()
    main(args)
