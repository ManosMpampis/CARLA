"""Stable tag paths, phase-one handoffs, and explicit run compatibility."""
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.config import DEFAULT_VERSION, create_config
from utils.experiment_suite import build_plan, checkpoint_path, experiment_dir
from experiments_psm_all import cli


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        env = root / 'env.yml'
        env.write_text(yaml.safe_dump({'root_dir': str(root / 'results')}))
        entries = [
            {'framework': 'ae', 'experiment_name': 'default', 'runner': 'ae',
             'config': str(REPO / 'configs/baselines/smd_ae.yml')},
            {'framework': 'lewm_encoder', 'experiment_name': 'time', 'runner': 'lewm',
             'config': str(REPO / 'configs/lewm_encoder/time/phase1.yml')},
            {'framework': 'cross_attention', 'experiment_name': 'observed_input',
             'runner': 'cross_attention',
             'config': str(REPO / 'configs/lewm_encoder/time/cross_attention/observed_input.yml'),
             'pretrained_experiment': 'lewm_encoder/time'},
        ]
        manifest = root / 'suite.yml'
        common = {'train_db_name': 'psm', 'val_db_name': 'psm', 'fname': 'psm',
                  'model_kwargs': {'in_channels': 25}}
        manifest.write_text(yaml.safe_dump({'common': common,
                                           'experiments': entries}))
        plan, output = build_plan(manifest, env, DEFAULT_VERSION)
        assert not Path(output).exists()
        for exp in plan:
            cfg = create_config(env, exp.config_path, 'psm', update_dictionary=exp.config)
            directory = Path(cfg['jepa_dir'])
            assert directory == experiment_dir(exp, output, DEFAULT_VERSION)
            assert cfg['version'] == DEFAULT_VERSION
            assert directory.parent.name == ('phase1' if exp.runner == 'lewm' else exp.name)
            again = create_config(env, exp.config_path, 'psm', update_dictionary=exp.config)
            assert again['jepa_checkpoint'] == cfg['jepa_checkpoint']
            tagged = create_config(env, exp.config_path, 'psm', update_dictionary={**exp.config, 'tag_jepa': 'another_tag'})
            assert tagged['jepa_dir'] != cfg['jepa_dir']
            named = create_config(env, exp.config_path, 'psm', '2026-10-11', exp.config)
            assert Path(named['jepa_dir']).parent.name == '2026-10-11'
            assert len(Path(named['jepa_dir']).parts) == len(directory.parts) + 1
        assert plan[-1].config['pretrained_from'] == str(checkpoint_path(plan[1], output, DEFAULT_VERSION))
        # An omitted encoder resolves from its stable path too.
        source = checkpoint_path(plan[1], output, DEFAULT_VERSION)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
        manifest.write_text(yaml.safe_dump({'common': common,
                                           'experiments': entries[2:]}))
        child, _ = build_plan(manifest, env, DEFAULT_VERSION)
        assert child[0].config['pretrained_from'] == str(source)
        for flags, expected in (([], DEFAULT_VERSION), (['--version', 'named'], 'named'),
                                (['--fresh'], None)):
            with patch.object(sys, 'argv', ['suite', '--dry-run', *flags]), \
                 patch('utils.suite_cli.build_plan', return_value=([], output)) as planner:
                cli()
                version = planner.call_args.args[2]
                assert version == expected if expected else version.startswith('run-')
                assert planner.call_args.kwargs['auto_resume'] is False
    print('Stable tag paths, repeat runs, explicit versions, source lookup and suite CLI: OK')


if __name__ == '__main__':
    main()
