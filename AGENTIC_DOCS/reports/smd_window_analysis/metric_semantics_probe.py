"""Read-only probes for the metric interpretation in metric_semantics.md."""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import metrics.vus.metrics as vus_metrics
from utils.reconstruction_baselines import calibration_at_threshold


def main():
    original = vus_metrics.generate_curve_t
    received = []

    def record_volume_argument(labels, scores, buffer):
        received.append(buffer)
        return None, None, None, None, None, None, 0.5, 0.5

    try:
        vus_metrics.generate_curve_t = record_volume_argument
        vus_metrics.get_range_vus_roc(
            np.arange(6, dtype=float), np.array([0, 0, 1, 1, 0, 0]), 10)
    finally:
        vus_metrics.generate_curve_t = original
    print(json.dumps({
        "requested_buffer": 10,
        "volume_received_buffer": received[0],
        "oracle_above_clean_max": calibration_at_threshold(np.array([1., 2., 3.]), 5.),
        "oracle_below_clean_min": calibration_at_threshold(np.array([1., 2., 3.]), 0.),
    }, indent=2))


if __name__ == "__main__":
    main()
