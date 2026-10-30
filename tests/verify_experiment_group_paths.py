"""Experiment groups isolate outputs, summaries, resume and phase-one handoffs."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.config import create_config, dataset_output_dir, experiment_base_dir
from utils.experiment_suite import (build_plan, checkpoint_path, experiment_dir,
                                    load_suite_manifest, resume_latest, run_suite)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        env = root / 'env.yml'
        env.write_text(yaml.safe_dump({'root_dir': str(root / 'results')}))
        manifest = REPO / 'configs/smd/logging_check_10e.yml'
        plan, output = build_plan(manifest, env, 'run1')
        group = Path(output) / 'smd/logging_check_10e'
        assert len(plan) == 5 and not group.exists()
        source = next(exp for exp in plan if exp.runner == 'lewm')
        expected_source = str(checkpoint_path(source, output, 'run1'))
        for exp in plan:
            assert exp.config['epochs'] == 10 and exp.config['fname'] == 'machine-1-1.txt'
            path = experiment_dir(exp, output, 'run1')
            assert path.is_relative_to(group)
            cfg = create_config(env, exp.config_path, exp.config['fname'], 'run1', exp.config)
            assert Path(cfg['jepa_dir']) == path
            if exp.dependency:
                assert cfg['pretrained_from'] == expected_source
                assert cfg['phase1_checkpoint'] == expected_source
            legacy = {**exp.config}
            legacy.pop('experiment_version')
            legacy_path = Path(experiment_base_dir(output, legacy, legacy['fname'], 'run1'))
            assert 'logging_check_10e' not in legacy_path.parts
        # Locate omitted phase one only within this group.
        path = Path(expected_source)
        path.touch()
        path.with_name("checkpoint.pth.tar").touch()
        document = load_suite_manifest(manifest)
        document['experiments'] = document['experiments'][3:]
        child_manifest = root / 'children.yml'
        child_manifest.write_text(yaml.safe_dump(document))
        children, _ = build_plan(child_manifest, env, 'run1')
        assert len(children) == 2
        assert all(exp.config['pretrained_from'] == expected_source for exp in children)
        # Newer sources in another group must not redirect automatic resume.
        other = replace(source, config={**source.config, 'experiment_version': 'other'})
        other_path = checkpoint_path(other, output, 'run9')
        other_path.parent.mkdir(parents=True)
        other_path.touch()
        other_path.with_name("checkpoint.pth.tar").touch()
        assert resume_latest(source, output, 'run1').run_version == 'run1'
        metric = {'calibrated': {'f1': .5}, 'oracle': {'f1': .7},
                  'vus_pr': .6, 'vus_roc': .8}
        report = {'evaluation': {mode: metric for mode in ('window','timeseries','non_overlap_window')},
                  'coverage': {'scored_timesteps': 100, 'total_timesteps': 100, 'output_window': 32}}
        with patch('utils.experiment_suite.score_experiment', return_value=report):
            rows = run_suite(plan, output, env, 'run1', score_only=True)
        assert all(row['status'] == 'completed' for row in rows)
        assert (group / 'summaries/run1/machine-1-1.txt/summary.json').is_file()
        assert not (Path(output) / 'smd/summaries').exists()
        for invalid in ('../other', '/tmp/other', ''):
            try:
                dataset_output_dir(output, {'train_db_name':'smd','experiment_version':invalid})
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid experiment_version accepted: {invalid!r}')
    print('Experiment group paths, handoffs, source lookup, resume and summaries: OK')


if __name__ == '__main__':
    main()
