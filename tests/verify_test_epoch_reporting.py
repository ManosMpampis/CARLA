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
    metric = {'calibrated': {'threshold': 0.5, 'f1_no_pa': 0.4,
                             'event_f1_no_pa': 0.3, 'calibration_quantile': 0.99},
              'oracle': {'threshold': 0.2, 'f1_no_pa': 0.7,
                         'event_f1_no_pa': 0.6, 'calibration_quantile': 0.8}, 'vus_pr': 0.8, 'vus_roc': 0.9}
    evaluation = {name: metric for name in ('window', 'timeseries', 'non_overlap_window')}
    dataset = SimpleNamespace(series=[], targets=[])
    with tempfile.TemporaryDirectory() as root:
        logger = Mock()
        config = {'jepa_dir': root}
        with patch.multiple(baselines, get_jepa_datasets=Mock(return_value=(dataset, dataset)),
                            JEPADataset=Mock(return_value=dataset),
                            evaluate=Mock(return_value=evaluation)):
            for epoch in (2, 4):
                assert baselines.evaluate_on_test(Mock(), config, 'cpu', logger, epoch) == 0.4
        path = Path(root, 'test_evaluations.jsonl')
        assert path.exists(), 'scheduled test evaluations have no persistent epoch history'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row['epoch'] for row in rows] == [2, 4]
        assert all(row['evaluation'] == evaluation for row in rows)
        logs = '\n'.join(call.args[0] for call in logger.log.call_args_list)
        for protocol in evaluation:
            assert protocol in logs, (protocol, logs)
        for name in ('event_f1_no_pa', 'vus_pr', 'vus_roc', 'oracle', 'calibrated', 'calibration_quantile'):
            assert name in logs, (name, logs)
        assert logger.scalar_summary.call_count == 2 * 3 * 10
    print('Per-epoch test history, complete console metrics and TensorBoard: OK')


if __name__ == '__main__':
    main()
