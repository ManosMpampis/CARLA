"""VUS honors the configured maximum buffer without doubling or PA overrides."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.metrics import combine_all_evaluation_scores
from metrics.vus.metrics import get_range_vus_roc
from utils.reconstruction_baselines import evaluate_from_scores


def main():
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.1, 0.7, 0.2])
    with patch('metrics.vus.metrics.metricor') as grader, patch(
            'metrics.vus.metrics.generate_curve_t',
            return_value=(None,) * 6 + (0.8, 0.7)) as curve:
        grader.return_value.RangeAUC.return_value = (0.8, 0.7, None, None, None)
        for bound in (None, 7, 43):
            expected = 100 if bound is None else bound
            if bound is None:
                get_range_vus_roc(scores, labels)
            else:
                get_range_vus_roc(scores, labels, bound)
            assert curve.call_args.args[2] == expected
            assert grader.return_value.RangeAUC.call_args.kwargs['window'] == expected

    metrics = dict(R_AUC_ROC=0.8, R_AUC_PR=0.7, VUS_ROC=0.8, VUS_PR=0.7)
    with patch('metrics.metrics.get_range_vus_roc', return_value=metrics) as vus:
        for bound in (None, 17):
            vus.reset_mock()
            pred = (scores > 0.5).astype(int)
            if bound is None:
                combine_all_evaluation_scores(pred, labels)
            else:
                combine_all_evaluation_scores(pred, labels, bound)
            assert [c.args[2] for c in vus.call_args_list] == [
                100 if bound is None else bound] * 2

    # Conversion to window-score units remains shared by training and suite scoring.
    starts = np.arange(len(labels))
    scored = dict(window_scores=scores, timeseries_scores=scores,
                  starts=starts, ends=starts + 1, cover_counts=np.ones(len(labels)))
    with patch('utils.reconstruction_baselines._evaluate_protocol',
               return_value={}) as protocol:
        for bound in (None, 37):
            protocol.reset_mock()
            config = dict(wsz=1, stride=10)
            if bound is not None:
                config['eval_window_size'] = bound
            evaluate_from_scores(scored, scored, labels, config)
            expected = 100 if bound is None else bound
            assert [c.args[-1] for c in protocol.call_args_list] == [
                max(1, expected // 10), expected]
    print('VUS configured/default bounds, score-unit conversion and PA consistency: OK')


if __name__ == '__main__':
    main()
