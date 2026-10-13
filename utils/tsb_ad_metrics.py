"""Lazy access to official TSB-AD metrics, independent of CARLA model imports.

This module intentionally works on Python 3.11+ so exported CARLA scores can be
evaluated in an upstream-supported environment without importing Torch or the
training configuration stack.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


METRIC_NAMES = (
    "AUC-PR",
    "AUC-ROC",
    "VUS-PR",
    "VUS-ROC",
    "Standard-F1",
    "PA-F1",
    "Event-based-F1",
    "R-based-F1",
    "Affiliation-F",
)


def _upstream_modules(tsb_ad_root: str | Path | None, *, needs_window: bool):
    """Import upstream lazily, optionally from an explicit source checkout."""
    import_root = None
    if tsb_ad_root is not None:
        import_root = Path(tsb_ad_root).expanduser().resolve()
        if import_root.name == "TSB_AD":
            import_root = import_root.parent
        metrics_file = import_root / "TSB_AD" / "evaluation" / "metrics.py"
        if not metrics_file.is_file():
            raise ValueError(
                "--tsb-ad-root must contain TSB_AD/evaluation/metrics.py: "
                + str(import_root)
            )
        loaded = sys.modules.get("TSB_AD")
        if loaded is not None:
            loaded_paths = getattr(loaded, "__path__", ())
            requested_package = import_root / "TSB_AD"
            if not any(Path(p).resolve() == requested_package for p in loaded_paths):
                raise RuntimeError(
                    "TSB_AD is already imported from a different location; "
                    "start a fresh process to use " + str(import_root)
                )
        sys.path.insert(0, str(import_root))
        importlib.invalidate_caches()
    try:
        metric_module = importlib.import_module("TSB_AD.evaluation.metrics")
        window_module = (
            importlib.import_module("TSB_AD.utils.slidingWindows")
            if needs_window else None
        )
    except ImportError as exc:
        raise RuntimeError(
            "Official TSB-AD metrics are unavailable. Install TSB-AD in a "
            "supported Python 3.11/3.12 environment, or supply --tsb-ad-root "
            "with its source checkout and dependencies. Saved scores can be "
            "evaluated later with evaluate_tsb_ad.py. Import failed: " + str(exc)
        ) from exc
    finally:
        if import_root is not None:
            sys.path.remove(str(import_root))
    return metric_module, window_module


def load_metrics(tsb_ad_root: str | Path | None = None):
    """Preflight dependencies and return official metrics and window modules."""
    return _upstream_modules(tsb_ad_root, needs_window=True)


def _vector(value, name: str, *, length: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.size == 0:
        raise ValueError(name + " must be a nonempty one-dimensional array")
    if length is not None and len(array) != length:
        raise ValueError(
            name + " must align with scores: " + str(len(array))
            + " != " + str(length)
        )
    if np.iscomplexobj(array):
        raise ValueError(name + " must contain real numeric values")
    try:
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + " must contain numeric values") from exc
    if not np.isfinite(array).all():
        raise ValueError(name + " must contain only finite values")
    return array


def json_values(value: Any) -> Any:
    """Convert NumPy metric values to JSON values; undefined numbers are null."""
    if isinstance(value, np.ndarray):
        return json_values(value.tolist())
    if isinstance(value, np.floating):
        return json_values(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.generic):
        return json_values(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_values(item) for item in value]
    return value


def evaluate_scores(
    scores,
    labels,
    first_channel,
    *,
    tsb_ad_root: str | Path | None = None,
    sliding_window: int | None = None,
    predictions=None,
) -> dict[str, Any]:
    """Evaluate aligned scores using the official TSB-AD implementation.

    With the default ``predictions=None``, the five threshold-dependent metrics
    use upstream's oracle thresholds. This matches Run_Detector_U/M.py. Explicit
    predictions opt into upstream's fixed-prediction metrics. Neither path
    changes the detector's scores, training, calibration, or frozen CARLA metrics.
    The inferred metric window is upstream's first-channel period estimate,
    independent of the detector context length.
    """
    score_array = _vector(scores, "scores")
    label_array = _vector(labels, "labels", length=len(score_array))
    first_array = _vector(first_channel, "first_channel", length=len(score_array))
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("labels must contain only 0 or 1")
    label_array = label_array.astype(np.int64)
    pred_array = None
    if predictions is not None:
        pred_array = _vector(predictions, "predictions", length=len(score_array))
        if not np.isin(pred_array, (0, 1)).all():
            raise ValueError("predictions must contain only 0 or 1")
        pred_array = pred_array.astype(np.int64)
    if sliding_window is not None and (
        isinstance(sliding_window, (bool, np.bool_))
        or not isinstance(sliding_window, (int, np.integer))
        or sliding_window < 1
    ):
        raise ValueError("sliding_window must be a positive integer")
    metric_module, window_module = _upstream_modules(
        tsb_ad_root, needs_window=sliding_window is None
    )
    if sliding_window is None:
        sliding_window = window_module.find_length_rank(
            first_array.reshape(-1, 1), rank=1
        )
    if not isinstance(sliding_window, (int, np.integer)) or sliding_window < 1:
        raise ValueError("Official TSB-AD returned an invalid metric window")
    result = metric_module.get_metrics(
        score_array,
        label_array,
        slidingWindow=int(sliding_window),
        pred=pred_array,
    )
    if not isinstance(result, dict):
        raise TypeError("Official TSB-AD get_metrics did not return a dictionary")
    return json_values(result)
