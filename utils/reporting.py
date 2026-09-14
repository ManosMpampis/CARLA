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
from torchmetrics.functional.classification.precision_recall_curve import precision_recall_curve, _binary_clf_curve, _binary_precision_recall_curve_update
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
        "point_VUS_ROC": float(metric_dict["VUS_ROC"]),
        "point_VUS_PR": float(metric_dict["VUS_PR"]),
        "point_R_AUC_ROC": float(metric_dict["R_AUC_ROC"]),
        "point_R_AUC_PR": float(metric_dict["R_AUC_PR"]),
    }


def window_means(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Per-window mean of a per-timestep array (comparable to probe windows)."""
    return np.array([values[s:e].mean() for s, e in zip(starts, ends)])


def series_from_dataset(dataset):
    """Dense series matrix backing a dataset (train-side calibration input)."""
    return np.asarray(dataset.series, dtype=np.float32)

def _best_f1_threshold(scores, targets):
    """Compute the best F1 threshold for a given set of scores and targets."""
    state = _binary_precision_recall_curve_update(torch.from_numpy(scores), torch.from_numpy(targets), None)
    fps, tps, thresholds = _binary_clf_curve(state[0], state[1], pos_label=1)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    if (state[1] == 0).all():  # all labels are negative, recall is undefined
        recall = torch.ones_like(recall)

    # need to call reversed explicitly, since including that to slice would
    # introduce negative strides that are not yet supported in pytorch
    precision = torch.cat([precision.flip(0), torch.ones(1, dtype=precision.dtype, device=precision.device)])
    recall = torch.cat([recall.flip(0), torch.zeros(1, dtype=recall.dtype, device=recall.device)])
    thresholds = thresholds.flip(0).detach().clone()
    try:
        f1_score = 2*precision*recall / (precision+recall)
        if torch.isnan(f1_score).any():
            f1_score = torch.nan_to_num(f1_score)   
    except ZeroDivisionError:
        f1_score = [0.0]
    best_f1_index = torch.argmax(f1_score)
    best_f1_threshold = thresholds[best_f1_index]
    best_f1 = f1_score[best_f1_index].item()
    return best_f1_threshold


def _channel_decision(channels, clean_channels, quantile, operator="or"):
    """Threshold each reconstructed output channel independently."""
    names = sorted(k for k in channels if k.startswith("signal/channel/"))
    thresholds = {
        name: float(np.quantile(clean_channels[name], quantile))
        for name in names
    }
    flags = np.stack([np.asarray(channels[name]) >= thresholds[name]
                      for name in names])
    decisions = np.all(flags, axis=0) if operator == "and" else np.any(flags, axis=0)
    return decisions.astype(int), thresholds


def _best_channel_decision(channels, targets, operator="or"):
    names = sorted(k for k in channels if k.startswith("signal/channel/"))
    thresholds = {}
    for name in names:
        threshold = _best_f1_threshold(np.asarray(channels[name]), targets)
        thresholds[name] = float(threshold)
    flags = np.stack([np.asarray(channels[name]) >= thresholds[name]
                      for name in names])
    decisions = np.all(flags, axis=0) if operator == "and" else np.any(flags, axis=0)
    return decisions.astype(int), thresholds


def _save_timeseries_plot(path, series, targets, predictions, scores):
    """Save the first test channel with ground truth and detected labels."""
    import matplotlib.pyplot as plt

    steps = np.arange(len(series))
    fig, axes = plt.subplots(3, 1, figsize=(18, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1]})
    axes[0].plot(steps, series[:, 0], linewidth=0.6, color="tab:blue")
    axes[0].set_ylabel("channel 0")
    axes[1].plot(steps, targets, drawstyle="steps-post", linewidth=0.8,
                 color="tab:red")
    axes[1].set_ylabel("target")
    axes[2].plot(steps, predictions, drawstyle="steps-post", linewidth=0.8,
                 color="tab:orange")
    axes[2].set_ylabel("detected")
    axes[2].set_xlabel("timestep")
    axes[0].set_title("Test timeseries reconstruction scoring")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

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
    weights_path = p["jepa_model"]
    if not os.path.exists(weights_path):
        weights_path = p["jepa_checkpoint"]
    Trainer.load_weights(weights_path, model, logger, strict=True)
    model.eval()

    scorer = Scorer(model, device)
    score_bs = int(p.get("score_batch_size", p.get("batch_size", 256)))
    calibrator = Calibrator(**p.get("calibration_kwargs",
                                    {"quantile": 0.995}))

    # --- train-side scoring (clean + injected probes): the ONLY calibration input
    from utils.common_config import get_jepa_datasets

    train_dataset, _ = get_jepa_datasets(p)
    # Honest option: calibrate on the held-out val TAIL of the train series
    # (still clean-train data, never test labels). Val windows are unseen
    # during optimization, so their score scale matches test-normal better
    # when the model fits the train windows tightly.
    cal_src = p.get("calibration_source", "train")
    if cal_src == "val" and hasattr(train_dataset, "val_series"):
        clean_series = np.asarray(train_dataset.val_series, dtype=np.float32)
    else:
        clean_series = series_from_dataset(train_dataset)
    clean_result = scorer.score_series(clean_series, p["wsz"], p["stride"],
                                       batch_size=score_bs)
    clean_channels = {"fused": clean_result["scores"], **clean_result["channels"]}
    channel_mode = bool(p.get("threshold_per_channel", False))
    channel_operator = str(p.get("threshold_channel_operator", "or")).lower()
    if channel_operator not in {"or", "and"}:
        raise ValueError("threshold_channel_operator must be 'or' or 'and'")

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
        "calibration_source": cal_src,
    })
    logger.log(f"Calibration saved to {p['calibration_path']} "
               f"(threshold {threshold:.6g}, fallback={calibrator.fallback})")

    # --- test-side scoring through the same path
    test_dataset = JEPADataset(p, train=False)
    test_series = series_from_dataset(test_dataset)
    targets = np.asarray(test_dataset.targets).astype(int)
    test_result = scorer.score_series(test_series, p["wsz"], p["stride"],
                                      batch_size=score_bs)
    test_channels = {"fused": test_result["scores"], **test_result["channels"]}
    fused_test = calibrator.fuse(test_channels)
    if channel_mode:
        pred_labels, channel_thresholds = _channel_decision(
            test_channels, clean_channels, calibrator.quantile, channel_operator)
        calibrator.save(p["calibration_path"], extra={
            "threshold_mode": f"per_channel_{channel_operator}",
            "thresholds_per_channel": channel_thresholds,
        })
    else:
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

    # report theoretical best threshold (fused-test quantile) for reference, but do not use it
    best_f1_threshold = _best_f1_threshold(fused_test, targets)

    if channel_mode:
        best_pred_labels, best_channel_thresholds = _best_channel_decision(
            test_channels, targets, channel_operator)
    else:
        best_pred_labels = (torch.from_numpy(fused_test) >= best_f1_threshold).numpy().astype(int)
    best_metric_dict = combine_all_evaluation_scores(best_pred_labels, targets, window_size)

    best_metrics = honest_metrics(best_metric_dict, fused_test, targets,
                            test_result["start_idxs"], test_result["end_idxs"])
    best_point_adjust = {
        key[3:]: float(value) for key, value in best_metric_dict.items()
        if key.startswith("pa_") and isinstance(value, (int, float))
    }
    report["best_possible/honest"] = best_metrics
    report["best_possible/point_adjust_comparability"] = best_point_adjust
    report["best_possible"] = {"threshold": float(threshold)}

    # mandatory no-training baseline: same scoring path with an untrained
    # model of the identical architecture
    baseline_model = build_model(p).to(device)
    baseline_model.eval()
    baseline_result = Scorer(baseline_model, device).score_series(
        test_series, p["wsz"], p["stride"], batch_size=score_bs)
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
    if p.get("save_timeseries_plot", True):
        plot_path = p.get("timeseries_plot_path",
                          os.path.join(p["jepa_dir"], "timeseries.png"))
        _save_timeseries_plot(plot_path, test_series, targets, pred_labels,
                              fused_test)
        report["timeseries_plot"] = plot_path
    with open(p["metrics_path"], "w") as f:
        json.dump(report, f, indent=2)

    step = 1
    for section, values in report.items():
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if isinstance(value, (int, float)):
                logger.scalar_summary(section, name, value, step)
    logger.metrics_summary("Full metric dictionary", {
        k: float(v) for k, v in metric_dict.items()
        if isinstance(v, (int, float))
    }, step)
    logger.log(f"Metrics: honest={honest}")
    logger.log(f"Scores written to {p['scores_path']}, report to {p['metrics_path']}")
    logger.finalize()
    return report
