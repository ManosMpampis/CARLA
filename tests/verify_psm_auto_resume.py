"""Fast, isolated assertions for automatic PSM checkpoint discovery and loading."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.experiment_suite import build_plan, experiment_dir, checkpoint_path, train_experiment
from utils.trainer import Trainer

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    env = root / 'env.yml'
    env.write_text(yaml.safe_dump({'root_dir': str(root / 'results')}))
    manifest = root / 'experiments.yml'
    manifest.write_text(yaml.safe_dump({'common': {'model_kwargs': {'in_channels': 25}}, 'experiments': [
        {'framework': 'ae', 'experiment_name': 'default', 'runner': 'ae',
         'config': str(REPO / 'configs/baselines/smd_ae.yml')},
        {'framework': 'lewm_encoder', 'experiment_name': 'time', 'runner': 'lewm',
         'config': str(REPO / 'configs/lewm_encoder/time/phase1.yml')},
        {'framework': 'cross_attention', 'experiment_name': 'context_input',
         'runner': 'cross_attention',
         'config': str(REPO / 'configs/lewm_encoder/time/cross_attention/context_input.yml'),
         'pretrained_experiment': 'lewm_encoder/time'},
    ]}))
    plan, output = build_plan(manifest, env, 'new', auto_resume=True)
    assert all(exp.run_version is None for exp in plan)
    assert not Path(output).exists(), 'discovery must remain read-only'
    for version, timestamp in [('older', 100), ('latest', 200)]:
        for exp in plan[:2]:
            directory = experiment_dir(exp, output, version)
            directory.mkdir(parents=True)
            filename = 'last.pth.tar' if exp.runner == 'ae' else 'checkpoint.pth.tar'
            state = directory / filename
            state.touch()
            os.utime(state, (timestamp, timestamp))
            selected = checkpoint_path(exp, output, version)
            selected.touch()
            os.utime(selected, (timestamp, timestamp))
    # Best weights in a newer empty run are not resumable training state.
    directory = experiment_dir(plan[0], output, 'weights_only')
    directory.mkdir(parents=True)
    checkpoint_path(plan[0], output, 'weights_only').touch()
    resumed, _ = build_plan(manifest, env, 'new', auto_resume=True)
    assert [exp.run_version for exp in resumed] == ['latest', 'latest', None]
    assert resumed[2].config['pretrained_from'] == str(checkpoint_path(resumed[1], output, 'new'))
    calls = []
    with patch('utils.experiment_suite._function', return_value=lambda arm, args, **kwargs: calls.append(args.version)):
        train_experiment(resumed[0], env, 'new')
    assert calls == ['latest'], 'training must receive the discovered run identity'
    explicit, _ = build_plan(manifest, env, 'explicit')
    assert all(exp.run_version is None for exp in explicit)
    data = yaml.safe_load(manifest.read_text())
    data['experiments'] = data['experiments'][2:]
    manifest.write_text(yaml.safe_dump(data))
    external, _ = build_plan(manifest, env, 'new', auto_resume=True)
    assert len(external) == 1
    assert external[0].config['pretrained_from'] == str(checkpoint_path(resumed[1], output, 'new'))
    fresh, _ = build_plan(manifest, env, 'fresh', reuse_saved_sources=True)
    assert fresh[0].run_version is None
    # Exercise the real loader with a full model/optimizer/scheduler state.
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    expected = model.weight.detach().clone()
    state = root / 'checkpoint.pth.tar'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': 6, 'next_epoch': 7,
                'best_val_loss': 0.25}, state)
    with torch.no_grad():
        model.weight.zero_()
    optimizer.param_groups[0]['lr'] = 99
    epoch, best = Trainer.resume({'jepa_checkpoint': str(state)}, model, optimizer,
                                 scheduler, SimpleNamespace(log=lambda message: None))
    assert epoch == 7 and best == 0.25
    assert torch.equal(model.weight, expected)
    assert optimizer.param_groups[0]['lr'] == 0.003
    assert optimizer.state and scheduler.last_epoch == 1
print('Automatic per-experiment resume, dependencies, fresh selection, and state restoration: OK')
