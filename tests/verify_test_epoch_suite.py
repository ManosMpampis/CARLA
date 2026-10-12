"""Train tiny temporary PSM runs and verify every variant's epoch report."""
from pathlib import Path
import json
import tempfile

import yaml
import torch

from verify_psm_suite import fixture_manifest, make_psm
from utils.config import DEFAULT_VERSION
from utils.experiment_suite import build_plan, run_suite, experiment_dir


def main():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    with tempfile.TemporaryDirectory() as root:
        directory = Path(root)
        env = directory / 'env.yml'
        env.write_text('root_dir: ' + str(directory / 'results') + '\n')
        make_psm(directory)
        manifest, value = fixture_manifest(directory)
        # Keep the tiny fixture's channel and stride lists consistent with
        # current configs, and deduplicate variants now present in the manifest.
        value['common']['model_kwargs']['enc_strides'] = [2, 2]
        # Production manifests may omit encoder entries to reuse saved runs.
        # The isolated fixture must train those sources in its temporary root.
        source_names = {entry['experiment_name'] for entry in value['experiments']
                        if entry['runner'] == 'lewm'}
        for dependency in sorted({entry['pretrained_experiment']
                                  for entry in value['experiments']
                                  if entry.get('pretrained_experiment')}):
            name = dependency.removeprefix('lewm_encoder/')
            if name not in source_names:
                value['experiments'].append({
                    'framework': 'lewm_encoder', 'experiment_name': name,
                    'runner': 'lewm',
                    'config': str(Path(__file__).resolve().parents[1] /
                                  'configs/lewm_encoder' / name / 'phase1.yml'),
                    'overrides': {'criterion_kwargs': {'sigreg_kwargs': {'num_slices': 4}}}})
        seen, entries = set(), []
        for entry in value['experiments']:
            key = (entry['framework'], entry['experiment_name'], entry.get('pretrained_experiment'))
            if key not in seen:
                entries.append(entry)
                seen.add(key)
        value['experiments'] = entries
        manifest.write_text(yaml.safe_dump(value))
        plan, results = build_plan(manifest, env, DEFAULT_VERSION)
        rows = run_suite(plan, results, env, DEFAULT_VERSION)
        failures = [(row['experiment_key'], row.get('error')) for row in rows
                    if row['status'] != 'completed']
        assert not failures, failures
        for exp in plan:
            out = experiment_dir(exp, results, DEFAULT_VERSION)
            history = [json.loads(line) for line in (out / 'test_evaluations.jsonl').read_text().splitlines()]
            assert [row['epoch'] for row in history] == [1], exp.key
            assert set(history[0]['evaluation']) == {'window', 'timeseries', 'non_overlap_window'}, exp.key
            report = json.loads((out / 'metrics.json').read_text())
            for evaluation in (history[0]['evaluation'], report['evaluation']):
                for procedure, metrics in evaluation.items():
                    assert metrics['calibrated']['calibration_quantile'] == value['common']['calibration_quantile'], (exp.key, procedure)
                    assert 0 <= metrics['oracle']['calibration_quantile'] <= 1, (exp.key, procedure)
        assert (Path(results) / 'psm/summaries/summary.json').is_file()
        print(f'All {len(plan)} experiment variants persist their scheduled test evaluations: OK')


if __name__ == '__main__':
    main()
