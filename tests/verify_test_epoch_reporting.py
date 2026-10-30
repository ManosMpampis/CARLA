"""Every scheduled test report is visible and persisted with its epoch."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import reconstruction_baselines as baselines


def main():
    metric = {'calibrated': {'threshold': 0.5, 'f1': 0.4,
                             'event_f1': 0.3, 'calibration_quantile': 0.99},
              'oracle': {'threshold': 0.2, 'f1': 0.7,
                         'event_f1': 0.6, 'calibration_quantile': 0.8,
                         'calibration_threshold': 0.2,
                         'calibration_threshold_in_range': True,
                         'clean_anomaly_fraction': 0.2}, 'vus_pr': 0.8, 'vus_roc': 0.9}
    evaluation = {name: metric for name in ('window', 'timeseries', 'non_overlap_window')}
    dataset = SimpleNamespace(series=[], targets=[])
    with tempfile.TemporaryDirectory() as root:
        logger = Mock()
        config = {'jepa_dir': root}
        with patch.multiple(baselines, get_jepa_datasets=Mock(return_value=(dataset, dataset)),
                            JEPADataset=Mock(return_value=dataset),
                            evaluate=Mock(return_value=evaluation)):
            for epoch in (2, 4):
                assert baselines.evaluate_on_test(Mock(), config, 'cpu', logger, epoch) == evaluation
        path = Path(root, 'test_evaluations.jsonl')
        assert path.exists(), 'scheduled test evaluations have no persistent epoch history'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row['epoch'] for row in rows] == [2, 4]
        assert all(row['evaluation'] == evaluation for row in rows)
        logs = '\n'.join(call.args[0] for call in logger.log.call_args_list)
        for section in ("Test_window", "Test_non_overlap", "Test_timeseries"):
            assert section in logs, (section, logs)
        for name in ('event_f1', 'vus_pr', 'vus_roc', 'oracle', 'calibrated', 'calibration_quantile'):
            assert name in logs, (name, logs)
        assert logger.scalar_summary.call_count == 2 * 3 * 10
        sections = {call.args[0].split('/')[0] for call in logger.scalar_summary.call_args_list}
        assert sections == {'Test_window', 'Test_non_overlap', 'Test_timeseries'}
        tags = {f"{call.args[0]}/{call.args[1]}" for call in logger.scalar_summary.call_args_list}
        assert 'Test_timeseries/oracle/threshold' in tags
        assert 'Test_timeseries/oracle/quantile' in tags
        assert 'Test_timeseries/oracle/f1' in tags
        assert not any('no_pa' in tag for tag in tags)
        names = {call.args[1] for call in logger.scalar_summary.call_args_list}
        assert not names & {'calibration_threshold', 'calibration_threshold_in_range',
                            'clean_anomaly_fraction'}, names
    print('Per-epoch test history, complete console metrics and TensorBoard: OK')


if __name__ == '__main__':
    main()
