"""Two-phase LEWM harness: latent pretraining, then mirrored reconstruction.

After LEWM pretraining through this entry point, the same
LeWMResNetEncoder (optionally strided via ``model_kwargs.enc_strides``)
is reused, and a NEW mirrored head (models/recon_head.py: N
ConvTranspose1d at the mirrored rates + final 1x1 conv to input channels)
is trained to reconstruct the normal input with mean L1 over C x W.

Default inference is encoder + head only (fully convolutional,
timestep-agnostic for any W with W % S == 0). The aux-crop variant keeps
the frozen pretext TimeAuxiliary to propose latent crops at inference
(see ReconModel.score); scoring uses configurable L1/L2/MSE reduction.
"""
import argparse
import os

import numpy as np
import torch

from models.builders import get_recon_model
from lewm import _device, run_pretrain, run_score as run_encoder_score, set_seed

from utils.common_config import (
    get_criterion,
    get_jepa_datasets,
    get_optimizer,
    get_scheduler,
    get_train_dataloader,
    get_val_dataloader,
)
from utils.config import create_config, entry_overrides
from utils.trainer import Trainer
from utils.utils import Logger


class _GraphWrapper(torch.nn.Module):
    """Tensor-only view for TensorBoard graph logging (full-window recon)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["recon"]


def _make_logger(p):
    logger = Logger(p["version"], verbose=2, file_path=p["jepa_dir"],
                    use_tensorboard=True, delete_files=False)
    logger.log(f"LeWM reconstruction stage '{p.get('stage', 'recon')}' --> ")
    logger.log_hyperparams(p)
    return logger


def _assert_divisible(p, model):
    s = int(getattr(model, "total_stride", 1))
    wsz = int(p["wsz"])
    if wsz % s != 0:
        raise ValueError(f"Phase-2 contract: wsz ({wsz}) must be divisible "
                         f"by total stride S={s} (enc_strides={p['model_kwargs'].get('enc_strides')})")


def _load_encoder_source(p, model, logger):
    """Load encoder (+aux when present) from the LeWM pretraining checkpoint."""
    source = p.get("pretrained_from")
    if not source or not os.path.exists(source):
        raise FileNotFoundError(
            f"recon requires 'pretrained_from' LeWM checkpoint; got {source}"
        )
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload["model"] if "model" in payload else payload
    # Transfer only the modules reused by phase two; the new head stays random.
    for name, module in (("encoder", model.encoder), ("aux", model.aux)):
        if module is None:
            continue
        prefix = f"{name}."
        weights = {key.removeprefix(prefix): value for key, value in state.items()
                   if key.startswith(prefix)}
        if not weights:
            raise ValueError(f"no {name} weights in LEWM checkpoint {source}")
        module.load_state_dict(weights, strict=True)
    logger.log(f"Recon init: encoder (+aux) loaded from {source}, head random")


def _build_run(p, device):
    model = get_recon_model(p)
    criterion = get_criterion(p).to(device)
    optimizer = get_optimizer(p, model)
    scheduler = get_scheduler(p, optimizer)
    return model, criterion, optimizer, scheduler


@torch.no_grad()
def eval_recon_on_test(model, p, device, logger, step):
    """Evaluate window means and overlap-averaged series with both thresholds."""

    from data.jepa_dataset import JEPADataset
    from utils.common_config import get_jepa_datasets
    from utils.reconstruction_baselines import evaluate

    flag = bool(getattr(model, "score_aux_crop", False))
    model.score_aux_crop = False
    try:
        _, val_dataset = get_jepa_datasets(p)
        test_dataset = JEPADataset(p, train=False)
        eval_config = dict(p)
        eval_config["calibration_quantile"] = float(
            p.get("calibration_kwargs", {}).get("quantile", 0.99))
        metrics = evaluate(model, val_dataset.series, test_dataset.series,
                           test_dataset.targets, eval_config, device)
        for procedure, values in metrics.items():
            for threshold_source in ("calibrated", "oracle"):
                for name, value in values[threshold_source].items():
                    logger.scalar_summary(f"test/{procedure}/{threshold_source}",
                                          name, value, step)
            logger.scalar_summary(f"test/{procedure}", "vus_pr",
                                  values["vus_pr"], step)
            logger.scalar_summary(f"test/{procedure}", "vus_roc",
                                  values["vus_roc"], step)
        logger.log(f"Test eval [{step}] window F1 "
                   f"{metrics['window']['calibrated']['f1_no_pa']:.4f}, "
                   f"time-series F1 "
                   f"{metrics['timeseries']['calibrated']['f1_no_pa']:.4f}")
    finally:
        model.score_aux_crop = flag
    return metrics["timeseries"]["calibrated"]["f1_no_pa"]

def run_recon(p, device):
    """Train the mirrored head on normal-only windows (dense L1)."""
    logger = _make_logger(p)
    model, criterion, optimizer, scheduler = _build_run(p, device)
    _assert_divisible(p, model)
    if not os.path.exists(p["jepa_checkpoint"]):
        _load_encoder_source(p, model, logger)

    mode = p.get("stage_c", {}).get("mode", "frozen")
    if mode == "frozen":
        for param in model.encoder.parameters():
            param.requires_grad = False
        if model.aux is not None:
            for param in model.aux.parameters():
                param.requires_grad = False
        model.encoder_frozen = True
        logger.log("Recon mode 'frozen': encoder (+aux) frozen, head trains")
    elif mode == "finetune":
        logger.log("Recon mode 'finetune': all parameters update")
    else:
        raise ValueError(f"Invalid stage_c.mode {mode}")

    model = model.to(device)
    optimizer = get_optimizer(p, model)
    scheduler = get_scheduler(p, optimizer)

    in_channels = p["model_kwargs"]["in_channels"]
    try:
        logger.add_graph(_GraphWrapper(model),
                         torch.rand((1, in_channels, p["wsz"]), device=device))
    except Exception as exc:
        logger.warn(f"TensorBoard graph logging skipped: {exc}")

    train_dataset, val_dataset = get_jepa_datasets(p)
    train_loader = get_train_dataloader(p, train_dataset)
    val_loader = get_val_dataloader(p, val_dataset)
    logger.log(f"Dataset contains {len(train_dataset)}/{len(val_dataset)} "
               f"train/val samples")

    # Dense recon: no masking collator (normal-only clean windows).
    trainer = Trainer(p, model, criterion, optimizer, scheduler, device, logger,
                      collator=None, amp=bool(p.get("amp", False)) and device.type == "cuda")
    start_epoch, best_val_loss = Trainer.resume(p, model, optimizer, scheduler, logger)
    # Monitoring-only test eval every X epochs (default 10): logs test/ to
    # TensorBoard + log.txt; checkpoint selection stays on val (no leakage).
    eval_every = int(p.get("stage_c", {}).get("eval_every", 10))
    best_val_loss = trainer.fit(
        train_loader, val_loader, start_epoch, best_val_loss,
        eval_every=eval_every,
        eval_fn=(lambda step: eval_recon_on_test(model, p, device, logger, step))
        if eval_every > 0 else None,
    )
    logger.log(f"Recon training finished; best val loss {best_val_loss:.6f}")
    logger.finalize()


@torch.no_grad()
def run_score(p, device):
    """Score both complete windows and overlap-averaged time series."""
    import json

    from data.jepa_dataset import JEPADataset
    from utils.reconstruction_baselines import score_both, evaluate_from_scores
    from utils.scoring import evaluation_options

    logger = _make_logger(p)
    model = get_recon_model(p)
    _assert_divisible(p, model)
    if bool(p.get("score_aux_crop", False)) and model.aux is None:
        raise ValueError("score_aux_crop=true needs recon_kwargs.with_aux=true "
                         "and aux weights in the recon checkpoint")
    weights = p.get("score_checkpoint") or (
        p["jepa_model_best"] if p.get("score_with_best_f1", False)
        else p["jepa_model"])
    if not os.path.exists(str(weights)):
        raise FileNotFoundError(f"reconstruction weights not found: {weights}")
    Trainer.load_weights(str(weights), model, logger, strict=True)
    model = model.to(device).eval()
    _, val_dataset = get_jepa_datasets(p)
    test_dataset = JEPADataset(p, train=False)
    options = evaluation_options(p)
    wsz, stride = options.pop("wsz"), options.pop("stride")
    batch_size = int(p.get("score_batch_size", p.get("batch_size", 256)))
    clean = score_both(model, val_dataset.series, wsz, stride,
                       batch_size, device, **options)
    test = score_both(model, test_dataset.series, wsz, stride,
                      batch_size, device, **options)
    eval_config = dict(p)
    quantile = float(p.get("calibration_kwargs", {}).get("quantile", 0.99))
    eval_config["calibration_quantile"] = quantile
    evaluation = evaluate_from_scores(clean, test, test_dataset.targets,
                                      eval_config)
    baseline_model = get_recon_model(p).to(device).eval()
    baseline_clean = score_both(baseline_model, val_dataset.series, wsz,
                                stride, batch_size, device, **options)
    baseline_test = score_both(baseline_model, test_dataset.series, wsz,
                               stride, batch_size, device, **options)
    baseline_evaluation = evaluate_from_scores(
        baseline_clean, baseline_test, test_dataset.targets, eval_config)
    with open(p["calibration_path"], "w") as stream:
        json.dump({"source": "held-out validation tail", "quantile": quantile,
                   "score_mode": model.score_mode,
                   "window_threshold": evaluation["window"]["calibrated"]["threshold"],
                   "timeseries_threshold": evaluation["timeseries"]["calibrated"]["threshold"]},
                  stream, indent=2)
    report = {"score_mode": model.score_mode,
              "framework": "reconstruction",
              "phase1_experiment": p.get("phase1_experiment"),
              "experiment_name": p.get("experiment_name"),
              "pretrained_from": p.get("pretrained_from"),
              "selection": {"weights": str(weights),
                            "threshold_source": "validation and test oracle, separate"},
              "evaluation": evaluation,
              "no_training_baseline": baseline_evaluation}
    with open(p["metrics_path"], "w") as stream:
        json.dump(report, stream, indent=2)
    np.savez_compressed(
        p["scores_path"],
        window_scores=test["window_scores"],
        timeseries_scores=test["timeseries_scores"],
        start_idxs=test["starts"], end_idxs=test["ends"],
        input_start_idxs=test["input_starts"],
        input_end_idxs=test["input_ends"],
        cover_counts=test["cover_counts"],
        window_predictions=(test["window_scores"] >
                            evaluation["window"]["calibrated"]["threshold"]),
        timeseries_predictions=(test["timeseries_scores"] >
                                evaluation["timeseries"]["calibrated"]["threshold"]),
        window_labels=np.asarray([np.any(test_dataset.targets[s:e])
                                  for s, e in zip(test["starts"], test["ends"])],
                                 dtype=np.int64),
        timestep_labels=np.asarray(test_dataset.targets, dtype=np.int64),
    )
    for procedure, values in evaluation.items():
        for source in ("calibrated", "oracle"):
            for metric, value in values[source].items():
                logger.scalar_summary(f"test/{procedure}/{source}", metric,
                                      value, 1)
        logger.scalar_summary(f"test/{procedure}", "vus_pr", values["vus_pr"], 1)
        logger.scalar_summary(f"test/{procedure}", "vus_roc", values["vus_roc"], 1)
    logger.log(f"Reconstruction score report: {p['metrics_path']}")
    logger.finalize()
    return report


STAGES = {
    "pretrain": run_pretrain,
    "pretext": run_pretrain,
    "phase1": run_pretrain,
    "recon": run_recon,
    "phase2": run_recon,
    "adapt": run_recon,
    "score": run_score,
}


def main(args, update_dictionary=None):
    overrides = entry_overrides(args, update_dictionary)
    p = create_config(args.config_env, args.config_exp, args.fname, args.version,
                      update_dictionary=overrides)
    set_seed(int(p.get("seed", 4)))
    stage = str(p.get("stage", "pretrain")).lower()
    if stage not in STAGES:
        raise ValueError(f"Invalid stage {stage}; expected one of {sorted(STAGES)}")
    handler = run_encoder_score if stage == "score" and p["criterion"] == "lewm" else STAGES[stage]
    return handler(p, _device(p))


def cli():
    parser = argparse.ArgumentParser(description="Two-phase LEWM reconstruction TSAD harness")
    parser.add_argument("--config_env", required=True, help="Config file for the environment")
    parser.add_argument("--config_exp", required=True, help="Config file for the experiment")
    parser.add_argument("--fname", help="File name of the dataset machine", default="")
    parser.add_argument("--version", help="Experiment version", type=str)
    parser.add_argument("--stage", choices=sorted(STAGES))
    parser.add_argument("--score", action="store_true", help="score saved weights using this experiment config")
    parser.add_argument("--pretrained_from")
    parser.add_argument("--phase1-version", dest="phase1_version", help="shared encoder run; defaults to --version")
    parser.add_argument("--score_checkpoint")
    main(parser.parse_args())


if __name__ == "__main__":
    cli()
