"""Two-phase LEWM harness: original pretraining, then crop cross attention."""

import argparse
import json
import os

import numpy as np
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
    logger = Logger(p["version"], verbose=2, file_path=p["jepa_dir"],
                    use_tensorboard=True, delete_files=False)
    logger.log(f"Crop cross-attention LEWM stage '{p['stage']}'")
    logger.log_hyperparams(p)
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
        best = trainer.fit(train_loader, val_loader, epoch, best)
        logger.log(f"Phase2 finished; best validation prediction loss {best:.6f}")
    finally:
        logger.finalize()


@torch.no_grad()
def run_score(p, device):
    from data.jepa_dataset import JEPADataset
    from metrics.metrics import combine_all_evaluation_scores
    from utils.reconstruction_baselines import evaluate_from_scores, score_both
    from utils.scoring import covered_evaluation_view

    model = get_cross_attention_model(p).to(device)
    weights = p.get("score_checkpoint") or p["jepa_model"]
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"phase2 scoring weights not found: {weights}")
    Trainer.load_weights(weights, model, strict=True)
    options = scoring_options(p, model)
    window, stride = options.pop("wsz"), options.pop("stride")
    _, val_dataset = get_jepa_datasets(p)
    if not hasattr(val_dataset, "series"):
        raise ValueError("score one machine at a time; shared training checkpoints are supported")
    test_dataset = JEPADataset(p, train=False)
    logger = _logger(p)
    try:
        batch_size = int(p.get("score_batch_size", p["batch_size"]))
        clean = score_both(model, val_dataset.series, window, stride,
                           batch_size, device, **options)
        test = score_both(model, test_dataset.series, window, stride,
                          batch_size, device, **options)
        quantile = float(p.get("calibration_kwargs", {}).get("quantile", 0.995))
        evaluation = evaluate_from_scores(
            clean, test, test_dataset.targets, {**p, "calibration_quantile": quantile})
        calibration = {"source": "held-out clean train tail", "quantile": quantile,
                       "target": model.target, "query_source": model.query_source,
                       "score_mode": model.score_mode, "target_input_range": options["check_mask"],
                       "window_threshold": evaluation["window"]["calibrated"]["threshold"],
                       "timeseries_threshold": evaluation["timeseries"]["calibrated"]["threshold"]}
        report = {"framework": p.get("framework", "lewm_cross_attention"), "task": model.get_extra_state(),
                  "phase1_experiment": p.get("phase1_experiment"),
                  "experiment_name": p.get("experiment_name"),
                  "pretrained_from": p.get("pretrained_from"),
                  "score_mode": model.score_mode,
                  "selection": {"weights": str(weights), "checkpoint": "validation prediction loss"},
                  "evaluation": evaluation}
        covered_scores, covered_labels, _, _ = covered_evaluation_view(
            test["timeseries_scores"], test_dataset.targets, test["starts"],
            test["ends"], test["cover_counts"])
        # The frozen metric engine point-adjusts its prediction array in
        # place. Give it a private array so saved detections stay honest.
        metric_dict = combine_all_evaluation_scores(
            (covered_scores > calibration["timeseries_threshold"]).astype(np.int64),
            covered_labels, int(p.get("eval_window_size", 100)))
        report["honest"] = {key: float(value) for key, value in metric_dict.items()
                            if not key.startswith("pa_")}
        report["point_adjust_comparability"] = {
            key.removeprefix("pa_"): float(value) for key, value in metric_dict.items()
            if key.startswith("pa_")}
        from utils.decision_trace import write_final_decision_trace

        trace_output = write_final_decision_trace(
            model, device, test_dataset.series, p,
            calibration["timeseries_threshold"],
            options={"wsz": window, "stride": stride, **options},
            threshold_operator="gt")
        if trace_output is not None:
            report["decision_trace"] = trace_output
        for path, payload in ((p["calibration_path"], calibration), (p["metrics_path"], report)):
            with open(path, "w") as stream:
                json.dump(payload, stream, indent=2)
        np.savez_compressed(
            p["scores_path"], window_scores=test["window_scores"],
            timeseries_scores=test["timeseries_scores"],
            start_idxs=test["starts"], end_idxs=test["ends"],
            input_start_idxs=test["input_starts"], input_end_idxs=test["input_ends"],
            cover_counts=test["cover_counts"],
            window_predictions=test["window_scores"] > calibration["window_threshold"],
            timeseries_predictions=(test["timeseries_scores"] > calibration["timeseries_threshold"])
                                  & (test["cover_counts"] > 0),
            window_labels=np.asarray([np.any(test_dataset.targets[s:e])
                                      for s, e in zip(test["starts"], test["ends"])], dtype=np.int64),
            timestep_labels=test_dataset.targets)
        for protocol, values in evaluation.items():
            for source in ("calibrated", "oracle"):
                for name, value in values[source].items():
                    logger.scalar_summary(f"test/{protocol}/{source}", name, value, 1)
            for name in ("vus_pr", "vus_roc"):
                logger.scalar_summary(f"test/{protocol}", name, values[name], 1)
        for section in ("honest", "point_adjust_comparability"):
            for name, value in report[section].items():
                logger.scalar_summary(f"test/{section}", name, value, 1)
        logger.log(f"Cross-attention report: {p['metrics_path']}")
        return report
    finally:
        logger.finalize()


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
