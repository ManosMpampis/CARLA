"""Shared score-reporting helpers for the training entries.

Both entries calibrate on train-side scores only, threshold the fused test
scores, and repackage the single metric engine
(metrics.metrics.combine_all_evaluation_scores) into honest headlines,
point-adjust comparability columns, and a no-training baseline. The helpers
here are the exact logic previously inline in the trunk-only entry, moved
so the full-LeWM entry reports identically.
"""
import json
import os

import numpy as np
import torch


def honest_metrics(metric_dict, scores, targets, starts, ends) -> dict:
    """Honest headline: point metrics without point adjustment plus
    window-level AUROC/AP (per-window max score vs any anomaly inside)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    win_scores = np.array([scores[s:e].max() for s, e in zip(starts, ends)])
    win_labels = np.array([targets[s:e].max() for s, e in zip(starts, ends)])
    return {
        "point_AUROC": float(roc_auc_score(targets, scores)),
        "point_AP": float(average_precision_score(targets, scores)),
        "window_AUROC": float(roc_auc_score(win_labels, win_scores)),
        "window_AP": float(average_precision_score(win_labels, win_scores)),
        "point_precision": float(metric_dict["precision"]),
        "point_recall": float(metric_dict["recall"]),
        "point_F1_no_PA": float(metric_dict["f1_score"]),
        "MCC": float(metric_dict["MCC"]),
    }


def window_means(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Per-window mean of a per-timestep array (comparable to probe windows)."""
    return np.array([values[s:e].mean() for s, e in zip(starts, ends)])


def series_from_dataset(dataset):
    """Dense series matrix backing a dataset (train-side calibration input)."""
    return np.asarray(dataset.series, dtype=np.float32)


@torch.no_grad()
def score_with_model(p, device, build_model, logger) -> dict:
    """Full score stage for any model exposing the Scorer contract.

    `build_model(p)` constructs the (untrained) architecture; weights load
    from score_checkpoint/jepa_model/jepa_checkpoint. Emits calibration,
    scores, and the metrics report identically for both entries, and
    returns the report dict.
    """
    from data.jepa_dataset import JEPADataset
    from metrics.metrics import combine_all_evaluation_scores
    from utils.scoring import Calibrator, Scorer
    from utils.trainer import Trainer

    model = build_model(p).to(device)
    weights_path = p.get("score_checkpoint") or p["jepa_model"]
    if not os.path.exists(weights_path):
        weights_path = p["jepa_checkpoint"]
    Trainer.load_weights(weights_path, model, logger, strict=True)
    model.eval()

    scorer = Scorer(model, device)
    calibrator = Calibrator(**p.get("calibration_kwargs",
                                    {"quantile": 0.995}))

    # --- train-side scoring (clean + injected probes): the ONLY calibration input
    from utils.common_config import get_jepa_datasets

    train_dataset, _ = get_jepa_datasets(p)
    clean_series = series_from_dataset(train_dataset)
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
        probe_maps = {
            "fused": probe_scores.pop("fused"),
            **probe_scores.pop("levels"),
            **{f"signal/{k}": v for k, v in probe_scores.pop("signals").items()},
        }
        # comparable statistics on both sides: per-window means
        probe_channels = {k: v.mean(axis=1) for k, v in probe_maps.items()}
        clean_window_starts = clean_result["start_idxs"]
        clean_window_ends = clean_result["end_idxs"]
        fit_clean_channels = {
            k: window_means(v, clean_window_starts, clean_window_ends)
            for k, v in clean_channels.items()
        }
    else:
        fit_clean_channels = clean_channels

    calibrator.fit(fit_clean_channels, probes=probe_channels)

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
    test_series = series_from_dataset(test_dataset)
    targets = np.asarray(test_dataset.targets).astype(int)
    test_result = scorer.score_series(test_series, p["wsz"], p["stride"])
    test_channels = {"fused": test_result["scores"], **test_result["channels"]}
    fused_test = calibrator.fuse(test_channels)

    pred_labels = (fused_test >= threshold).astype(int)
    window_size = int(p.get("eval_window_size", 100))
    metric_dict = combine_all_evaluation_scores(pred_labels, targets, window_size)

    honest = honest_metrics(metric_dict, fused_test, targets,
                            test_result["start_idxs"], test_result["end_idxs"])
    point_adjust = {
        key[3:]: float(value) for key, value in metric_dict.items()
        if key.startswith("pa_") and isinstance(value, (int, float))
    }
    report = {"honest": honest, "point_adjust_comparability": point_adjust}

    # mandatory no-training baseline: same scoring path with an untrained
    # model of the identical architecture
    baseline_model = build_model(p).to(device)
    baseline_model.eval()
    baseline_result = Scorer(baseline_model, device).score_series(
        test_series, p["wsz"], p["stride"])
    baseline_fused = baseline_result.pop("scores")
    report["no_training_baseline"] = honest_metrics(
        combine_all_evaluation_scores((baseline_fused >= threshold).astype(int),
                                      targets, window_size),
        baseline_fused, targets,
        baseline_result["start_idxs"], baseline_result["end_idxs"])

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
    return report
