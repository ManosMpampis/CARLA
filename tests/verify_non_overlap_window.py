"""Verify disjoint scoring and window labels for full-window and crop heads."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.reconstruction_baselines import evaluate_non_overlap_window


class PositionScores:
    def __init__(self, crop=False):
        self.crop = crop

    def eval(self):
        return self

    def score(self, x):
        return {"fused": x[:, 0, 4:] if self.crop else x[:, 0]}


def main():
    series = np.arange(25, dtype=np.float32)[:, None]
    labels = np.zeros(25, dtype=np.int64)
    labels[[7, 8, 24]] = 1
    config = {"batch_size": 2, "calibration_quantile": 0.9, "eval_window_size": 16}
    for crop in (False, True):
        options = {"wsz": 8, "stride": 1, "include_last_window": True,
                   "check_mask": [4, 8] if crop else None, "latency_offset": 0}
        with patch('utils.reconstruction_baselines._evaluate_protocol', return_value={}) as evaluate:
            _, clean, test, stride = evaluate_non_overlap_window(
                PositionScores(crop), series, series, labels, config,
                torch.device('cpu'), options=options)
        assert options['stride'] == 1 and options['include_last_window'] is True
        assert stride == (4 if crop else 8)
        expected_starts = np.arange(4 if crop else 0, 24, stride)
        np.testing.assert_array_equal(test['starts'], expected_starts)
        np.testing.assert_array_equal(test['ends'], expected_starts + stride)
        for scored in (clean, test):
            assert scored['cover_counts'].max() == 1
            assert scored['cover_counts'][24] == 0  # incomplete tail is not rescored
            np.testing.assert_allclose(scored['window_scores'], expected_starts + (stride - 1) / 2)
            np.testing.assert_array_equal(scored['timeseries_scores'][scored['cover_counts'] > 0],
                                          series[expected_starts[0]:24, 0])
        args = evaluate.call_args.args
        expected_labels = [1, 1, 0, 0, 0] if crop else [1, 1, 0]
        np.testing.assert_array_equal(args[1], expected_labels)
        assert args[3] == 0.9
        assert args[4] == 16 // stride
    print('Non-overlap full/crop scoring: stride, counts, labels, clean calibration and tail: OK')


if __name__ == '__main__':
    main()
