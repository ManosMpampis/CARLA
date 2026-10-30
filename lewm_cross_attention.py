"""Two-phase LEWM harness: original pretraining, then crop cross attention."""

import argparse
import os

import torch

from models.builders import get_cross_attention_model

from lewm import _device, run_pretrain, run_score as run_encoder_score, set_seed
from utils.common_config import (get_criterion, get_jepa_datasets, get_optimizer,
                                 get_scheduler, get_train_dataloader,
                                 get_val_dataloader)
from utils.config import create_config, entry_overrides
from utils.masking_steered import SubAnomalyMaskCollator
from utils.trainer import Trainer
from utils.utils import Logger


def load_feature_extractor(source, model, logger=None):
    """Strict encoder-only transfer from either saved weights or a Trainer checkpoint."""
    if not source or not os.path.isfile(source):
        raise FileNotFoundError(f"phase2 requires pretrained_from LEWM weights; got {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload["model"] if "model" in payload else payload
    encoder_state = {key.removeprefix("encoder."): value
                     for key, value in state.items() if key.startswith("encoder.")}
    if not encoder_state:
        raise ValueError(f"no encoder weights in LEWM checkpoint {source}")
    model.encoder.load_state_dict(encoder_state, strict=True)
    if model.encoder_frozen:
        model.encoder.eval()
    if logger is not None:
        logger.log(f"Feature extractor loaded strictly from {source} "
                   f"(frozen={model.encoder_frozen}); Q/K/V and head are new")


def scoring_options(p, model):
    """Place residuals on the second crop; never calibrate padded context zeros."""
    from utils.scoring import evaluation_options

    options = evaluation_options(p)
    span = list(model.target_input_range(options["wsz"]))
    if options["check_mask"] is not None and list(options["check_mask"]) != span:
        raise ValueError(f"evaluation.check_mask must match the target crop {span}")
    if options["latency_offset"] != 0:
        raise ValueError("cross-attention scores already refer to the target crop; latency_offset must be 0")
    options["check_mask"] = span
    return options


def _logger(p):
    logger = Logger(p["version"], verbose=3, file_path=p["jepa_dir"],
                    use_tensorboard=True, delete_files=False)
    logger.log(f"Crop cross-attention LEWM stage '{p['stage']}'")
    return logger


def _collator(p):
    masking = dict(p.get("stage_b", {}).get("masking", {}))
    mode = masking.pop("mode", "none")
    if mode == "none":
        return None
    if mode != "subanomaly":
        raise ValueError("stage_b.masking.mode must be none or subanomaly")
    return SubAnomalyMaskCollator(**masking)


def run_phase2(p, device):
    if p["criterion"] != "lewm_cross_attention":
        raise ValueError("phase2 requires criterion=lewm_cross_attention")
    model = get_cross_attention_model(p).to(device)
    model.crop_ranges(int(p["wsz"]))
    logger = _logger(p)
    try:
        # A resumed run contains the complete frozen encoder and needs no
        # original phase-one file to still be available.
        if not os.path.exists(p["jepa_checkpoint"]):
            load_feature_extractor(p.get("pretrained_from"), model, logger)
        criterion = get_criterion(p).to(device)
        optimizer = get_optimizer(p, model)
        scheduler = get_scheduler(p, optimizer)
        train_dataset, val_dataset = get_jepa_datasets(p)
        train_loader = get_train_dataloader(p, train_dataset)
        val_loader = get_val_dataloader(p, val_dataset)
        if not len(train_loader) or not len(val_loader):
            raise ValueError("empty train/validation loader; reduce window or batch size")
        logger.log(f"Dataset contains {len(train_dataset)}/{len(val_dataset)} train/val windows")
        trainer = Trainer(p, model, criterion, optimizer, scheduler, device, logger,
                          collator=_collator(p),
                          amp=bool(p.get("amp", False)) and device.type == "cuda",
                          val_collator=_collator(p))
        epoch, best = Trainer.resume(p, model, optimizer, scheduler, logger)
        from utils.evaluation_schedule import evaluation_interval
        from utils.reconstruction_baselines import evaluate_on_test

        interval = evaluation_interval(p)
        best = trainer.fit(
            train_loader, val_loader, epoch, best, eval_every=interval,
            eval_fn=(lambda step: evaluate_on_test(
                model, p, device, logger, step, options=scoring_options(p, model)))
            if interval > 0 else None)
        logger.log(f"Phase2 finished; best validation prediction loss {best:.6f}")
    finally:
        logger.finalize()


@torch.no_grad()
def run_score(p, device):
    """Export the same three-mode report as every experiment runner."""
    from utils.experiment_suite import score_config
    return score_config(p, "cross_attention")


STAGES = {"pretrain": run_pretrain, "pretext": run_pretrain, "phase1": run_pretrain,
          "phase2": run_phase2, "adapt": run_phase2, "score": run_score}


def main(args, update_dictionary=None):
    overrides = entry_overrides(args, update_dictionary)
    p = create_config(args.config_env, args.config_exp, args.fname, args.version,
                      update_dictionary=overrides)
    set_seed(int(p.get("seed", 4)))
    stage = str(p.get("stage", "pretrain")).lower()
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}; got {stage}")
    handler = run_encoder_score if stage == "score" and p["criterion"] == "lewm" else STAGES[stage]
    return handler(p, _device(p))


def cli():
    parser = argparse.ArgumentParser(description="Two-phase convolutional cross-attention LEWM")
    parser.add_argument("--config_env", required=True)
    parser.add_argument("--config_exp", required=True)
    parser.add_argument("--fname", default="machine-1-1.txt")
    parser.add_argument("--version")
    parser.add_argument("--stage", choices=sorted(STAGES))
    parser.add_argument("--score", action="store_true", help="score saved weights using this experiment config")
    parser.add_argument("--pretrained_from")
    parser.add_argument("--phase1-version", dest="phase1_version", help="shared encoder run; defaults to --version")
    parser.add_argument("--score_checkpoint")
    main(parser.parse_args())


if __name__ == "__main__":
    cli()
