"""Phase-2 reconstruction training entry (grilled recon head).

Step after the steered pretext (carla_steered.py): the SAME
SteeredResNetEncoder (optionally strided via ``model_kwargs.enc_strides``)
is reused, and a NEW mirrored head (models/recon_head.py: N
ConvTranspose1d at the mirrored rates + final 1x1 conv to input channels)
is trained to reconstruct the normal input with mean L1 over C x W.

Default inference is encoder + head only (fully convolutional,
timestep-agnostic for any W with W % S == 0). The aux-crop variant keeps
the frozen pretext TimeAuxiliary to propose latent crops at inference
(see ReconModel.score); scoring is L1 recon error either way and the
threshold is the per-machine clean-train 0.99 quantile (calibration.json).
"""
import argparse
import os
import random

import numpy as np
import torch

from utils.common_config import (
    get_criterion,
    get_jepa_datasets,
    get_optimizer,
    get_scheduler,
    get_train_dataloader,
    get_val_dataloader,
)
from utils.config import create_config
from utils.trainer import Trainer
from utils.utils import Logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_recon_model(p):
    """Steered encoder + mirrored recon head (+ optional frozen aux)."""
    from models import get_backbone
    from models.recon_head import build_mirrored_head
    from models.recon_model import ReconModel
    from models.steered_lewm import TimeAuxiliary

    enc_kwargs = dict(p.get("model_kwargs", {}))
    built = get_backbone(p.get("backbone", "steered_resnet"), **enc_kwargs)
    encoder = built["model"]
    recon_kwargs = dict(p.get("recon_kwargs", {}))
    head = build_mirrored_head(
        encoder,
        norm=recon_kwargs.get("norm", enc_kwargs.get("norm", "batch")),
        dropout=float(recon_kwargs.get("dropout", enc_kwargs.get("dropout", 0.1))),
    )
    aux = None
    if bool(recon_kwargs.get("with_aux", False)):
        aux_kwargs = dict(p.get("aux_kwargs", {}))
        aux = TimeAuxiliary(
            int(encoder.output_dims),
            aux_channels=tuple(aux_kwargs.get("aux_channels", (32, 32, 32))),
            kernels=tuple(aux_kwargs.get("kernels", (7, 5, 3))),
            norm=enc_kwargs.get("norm", "batch"),
            dropout=enc_kwargs.get("dropout", 0.1),
        )
    model = ReconModel(encoder=encoder, head=head, aux=aux)
    # Scorer path (utils/reporting.score_with_model) constructs via this
    # builder, so the flag must live on the model itself.
    model.score_aux_crop = bool(p.get("score_aux_crop", False))
    return model


class _GraphWrapper(torch.nn.Module):
    """Tensor-only view for TensorBoard graph logging (full-window recon)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["recon"]


def _device(p):
    want = str(p.get("device", "cpu")).lower()
    if want.startswith("cuda") and not torch.cuda.is_available():
        want = "cpu"
    return torch.device(want)


def _make_logger(p):
    destructive = str(p.get("stage", "recon")).lower() != "score"
    logger = Logger(p["version"], verbose=2, file_path=p["jepa_dir"],
                    use_tensorboard=True, delete_files=destructive)
    logger.log(f"CARLA recon stage '{p.get('stage', 'recon')}' --> ")
    logger.log_hyperparams(p)
    return logger


def _assert_divisible(p, model):
    s = int(getattr(model, "total_stride", 1))
    wsz = int(p["wsz"])
    if wsz % s != 0:
        raise ValueError(f"Phase-2 contract: wsz ({wsz}) must be divisible "
                         f"by total stride S={s} (enc_strides={p['model_kwargs'].get('enc_strides')})")


def _load_encoder_source(p, model, logger):
    """Load encoder (+aux when present) from the steered pretext checkpoint."""
    source = p.get("pretrained_from")
    if not source or not os.path.exists(source):
        raise FileNotFoundError(
            f"recon requires 'pretrained_from' steered checkpoint; got {source}"
        )
    # strict=False: pretext predictor/mask_token ignored, new head kept random.
    Trainer.load_weights(source, model, logger, strict=False)
    logger.log(f"Recon init: encoder (+aux) loaded from {source}, head random")


def _build_run(p, device):
    model = get_recon_model(p)
    criterion = get_criterion(p).to(device)
    optimizer = get_optimizer(p, model)
    scheduler = get_scheduler(p, optimizer)
    return model, criterion, optimizer, scheduler


@torch.no_grad()
def eval_recon_on_test(model, p, device, logger, step):
    """Monitoring-only test eval for recon training (full-window mode).

    Scores the clean-train series for the 0.99 train-only threshold, then
    the test series, and logs test L1 + honest detection metrics under
    ``test/`` (TensorBoard + log.txt). Never touches checkpoint selection:
    best-model saving stays on the train-tail val loss, so test labels
    cannot steer training -- they are only observed here.
    """
    import numpy as np

    from data.jepa_dataset import JEPADataset
    from metrics.metrics import combine_all_evaluation_scores
    from utils.common_config import get_jepa_datasets
    from utils.reporting import honest_metrics, series_from_dataset
    from utils.scoring import Calibrator, Scorer

    flag = bool(getattr(model, "score_aux_crop", False))
    model.score_aux_crop = False
    try:
        train_dataset, _ = get_jepa_datasets(p)
        clean_series = series_from_dataset(train_dataset)
        test_dataset = JEPADataset(p, train=False)
        test_series = series_from_dataset(test_dataset)
        targets = np.asarray(test_dataset.targets).astype(int)

        scorer = Scorer(model, device)
        score_bs = int(p.get("score_batch_size", p.get("batch_size", 256)))
        clean = scorer.score_series(clean_series, p["wsz"], p["stride"],
                                    batch_size=score_bs)["scores"]
        calibrator = Calibrator(**p.get("calibration_kwargs", {"quantile": 0.99}))
        calibrator.fit({"fused": clean})
        threshold = calibrator.threshold_for(calibrator.fuse({"fused": clean}))

        test_result = scorer.score_series(test_series, p["wsz"], p["stride"],
                                            batch_size=score_bs)
        test = test_result["scores"]
        pred = (test >= threshold).astype(int)
        window_size = int(p.get("eval_window_size", 100))
        metric_dict = combine_all_evaluation_scores(pred, targets, window_size)
        honest = honest_metrics(metric_dict, test, targets,
                                test_result["start_idxs"], test_result["end_idxs"])
        logger.scalar_summary("test", "l1", float(np.mean(test)), step)
        logger.scalar_summary("test", "threshold", float(threshold), step)
        logger.scalar_summary("test", "point_AUROC", honest["point_AUROC"], step)
        logger.scalar_summary("test", "point_AP", honest["point_AP"], step)
        logger.scalar_summary("test", "point_F1_no_PA", honest["point_F1_no_PA"], step)
        logger.scalar_summary("test", "window_AUROC", honest["window_AUROC"], step)
        logger.scalar_summary("test", "window_AP", honest["window_AP"], step)
        logger.log(f"Test eval [{step}]: l1 {np.mean(test):.6f} thr {threshold:.6g} "
                   f"AUROC {honest['point_AUROC']:.4f} AP {honest['point_AP']:.4f} "
                   f"F1 {honest['point_F1_no_PA']:.4f}")
    finally:
        model.score_aux_crop = flag


def run_recon(p, device):
    """Train the mirrored head on normal-only windows (dense L1)."""
    logger = _make_logger(p)
    model, criterion, optimizer, scheduler = _build_run(p, device)
    _assert_divisible(p, model)
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
                      collator=None, amp=p.get("amp", False))
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
    """Score stage: shared engine, per-machine train-only 0.99 threshold."""
    from utils.reporting import score_with_model

    logger = _make_logger(p)
    model_probe = get_recon_model(p)
    _assert_divisible(p, model_probe)
    if bool(p.get("score_aux_crop", False)) and model_probe.aux is None:
        raise ValueError("score_aux_crop=true needs recon_kwargs.with_aux=true "
                         "and aux weights in the recon checkpoint")
    # Honor an explicit weights pointer (score configs document it as
    # score_checkpoint); otherwise the shared engine scores this run dir.
    override = p.get("score_checkpoint")
    if override and os.path.exists(str(override)):
        p["jepa_model"] = str(override)
        p["jepa_checkpoint"] = str(override)
    score_with_model(p, device, get_recon_model, logger)


STAGES = {
    "recon": run_recon,
    "pretext": run_recon,
    "pretrain": run_recon,
    "adapt": run_recon,
    "score": run_score,
}


def main(args, update_dictionary={}):
    p = create_config(args.config_env, args.config_exp, args.fname, args.version,
                      update_dictionary=update_dictionary)
    set_seed(int(p.get("seed", 4)))
    stage = str(p.get("stage", "recon")).lower()
    if stage not in STAGES:
        raise ValueError(f"Invalid stage {stage}; expected one of {sorted(STAGES)}")
    STAGES[stage](p, _device(p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase-2 recon TSAD harness")
    parser.add_argument("--config_env", help="Config file for the environment")
    parser.add_argument("--config_exp", help="Config file for the experiment")
    parser.add_argument("--fname", help="File name of the dataset machine", default="")
    parser.add_argument("--version", help="Experiment version", type=str)
    args = parser.parse_args()
    main(args)
