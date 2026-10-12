"""Inverse calibration checks, including ties and unattainable thresholds."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.reconstruction_baselines import calibration_at_threshold, _evaluate_protocol


def main():
    for scores, threshold, expected_q, in_range in (
        ([0., 10., 20.], 5., 0.25, True),
        ([0., 10., 10., 20.], 10., 2 / 3, True),
        ([0., 10., 20.], -1., 0., False),
        ([0., 10., 20.], 21., 1., False),
        ([7.], 7., 0., True),
        ([7.], 8., 0., False),
        ([7., 7., 7.], 7., 1., True),
        ([0., 10., 20.], np.nextafter(10., -np.inf),
         np.nextafter(10., -np.inf) / 20, True),
    ):
        result = calibration_at_threshold(scores, threshold)
        np.testing.assert_allclose(result["calibration_quantile"], expected_q)
        assert result["calibration_threshold_in_range"] == in_range
        expected_threshold = np.clip(threshold, min(scores), max(scores))
        np.testing.assert_allclose(result["calibration_threshold"], expected_threshold)
        assert result["clean_anomaly_fraction"] == np.mean(np.asarray(scores) > threshold)
    # Match the actual default np.quantile method over many interpolated values.
    rng = np.random.default_rng(4)
    scores = rng.normal(size=100)
    for q in np.linspace(0, 1, 101):
        threshold = np.quantile(scores, q)
        result = calibration_at_threshold(scores, threshold)
        np.testing.assert_allclose(result["calibration_quantile"], q, atol=1e-14)
        np.testing.assert_allclose(result["calibration_threshold"], threshold, atol=1e-14)
    clean = np.asarray([0., 10., 20.])
    with patch('utils.reconstruction_baselines.get_range_vus_roc',
               return_value={'VUS_PR': 0.5, 'VUS_ROC': 0.5}):
        result = _evaluate_protocol(np.asarray([1., 5., 15., 18.]),
                                    np.asarray([0, 0, 1, 1]), clean, 0.95, 1)
    assert result['calibrated']['calibration_quantile'] == 0.95
    np.testing.assert_allclose(result['calibrated']['threshold'], np.quantile(clean, 0.95))
    expected = calibration_at_threshold(clean, result['oracle']['threshold'])
    assert all(result['oracle'][name] == value for name, value in expected.items())
    print("Oracle calibration quantile, interpolation, ties, range and anomaly fraction: OK")


if __name__ == "__main__":
    main()
