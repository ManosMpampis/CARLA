"""Metric isolation, three-mode logging and bounded final figures."""
from pathlib import Path
import base64
import csv
import re
import json
import sys
import tempfile
from unittest.mock import Mock, patch

import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.evaluation_metrics import decision_metrics, score_metrics
from utils.reconstruction_baselines import _evaluate_protocol
from utils.evaluation_logging import (MODES, log_evaluation, record_evaluation,
                                      truncate_epoch_history)
from utils.evaluation_artifacts import write_score_html


def main():
    labels = np.array([0,1,1,1,0,0,0,1,1,0])
    scores = np.array([.1,.2,.9,.2,.8,.1,.3,.7,.2,.1])
    original = scores.copy()
    metrics = decision_metrics(scores, labels, .5)
    assert metrics['f1'] == f1_score(labels, scores>.5)
    assert metrics['pa_f1'] > metrics['f1']
    assert list(metrics)[-1] == 'pa_f1'
    assert decision_metrics(np.array([0.,1.,0.]), np.array([1,1,0]), .5)['pa_f1'] == 1.0
    np.testing.assert_array_equal(scores, original)
    # A normal point immediately after a ground-truth event is not a hit.
    boundary = decision_metrics(np.array([0.,0.,1.,0.]), np.array([0,1,0,0]), .5)
    assert boundary['event_f1'] == 0
    independent = score_metrics(scores, labels, 2)
    precision, recall, _ = precision_recall_curve(labels,scores)
    assert independent['auc_pr'] == auc(recall,precision)
    assert independent['auc_roc'] == roc_auc_score(labels,scores)
    for labels_ in (np.zeros(10,dtype=int),np.ones(10,dtype=int)):
        assert all(v is None for v in score_metrics(scores,labels_,2).values())
        assert np.isfinite(decision_metrics(scores,labels_,.5)['pa_f1'])
    values = _evaluate_protocol(scores, labels, np.linspace(.1,.9,20), .99, 2)
    evaluation = {mode: values for mode in MODES}
    writer = Mock()
    log_evaluation(writer,evaluation,4)
    tags = [call.args[0] for call in writer.add_scalar.call_args_list]
    assert len(tags) == len(set(tags))
    assert {tag.split('/')[0] for tag in tags} == set(MODES.values())
    with tempfile.TemporaryDirectory() as root:
        logs = []
        for epoch in (2,4):
            record_evaluation({'jepa_dir':root},evaluation,epoch,logs.append)
        assert len(logs) == 6
        for section in MODES.values():
            with Path(root,f'{section}.csv').open() as stream:
                reader = csv.DictReader(stream); rows = list(reader)
                assert reader.fieldnames[-2:] == ['calibrated_pa_f1','oracle_pa_f1']
                assert [r['epoch'] for r in rows] == ['2','4']
                assert not any('threshold_in_range' in k for k in reader.fieldnames)
        truncate_epoch_history(root, 2)
        record_evaluation({'jepa_dir':root},evaluation,4,logs.append)
        for section in MODES.values():
            with Path(root,f'{section}.csv').open() as stream:
                assert [r['epoch'] for r in csv.DictReader(stream)] == ['2','4']
        history = [json.loads(line) for line in Path(root,'test_evaluations.jsonl').read_text().splitlines()]
        assert [row['epoch'] for row in history] == [2,4]
        # Same input length/window scale as the 64 MB legacy replay repro.
        long_scores = np.zeros(8448);long_scores[4131]=100
        path = Path(root,'figure.html')
        write_score_html(path,long_scores,np.zeros(8448),1.,'Test_timeseries')
        assert path.stat().st_size < 500_000
        html = path.read_text()
        encoded = re.search(r"const encodedTrace = '([^']+)'", html).group(1)
        trace = json.loads(base64.b64decode(encoded))
        values = np.frombuffer(base64.b64decode(trace['final_scores_base64']), dtype='<f8')
        assert len(values) == 8448 and values[4131] == 100
        assert trace['checking_windows'] == [[0,8448,0,8448]]
        assert 'ResizeObserver' not in html and 'setInterval' not in html
        assert '<svg' in html and 'decision-y-scale' in html
        assert 'value="symlog" selected' in html
        assert 'decision-series-zoom-in' in html and "'pointermove'" in html
        assert path.stat().st_size < 500_000
        print(f'Full-resolution HTML: {path.stat().st_size/1e3:.1f} KB (legacy repro: 64 MB)')
    print('Metrics, PA isolation/order, split logs/CSVs and figure size: OK')


if __name__ == '__main__':
    main()
