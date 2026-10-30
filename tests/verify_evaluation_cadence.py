"""Real snapshot writes, cadence, independent best selections and resume."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from utils.trainer import Trainer
from utils.checkpointing import Checkpoints
from utils.evaluation_schedule import evaluation_interval, should_evaluate


def main():
    assert evaluation_interval({}) == 10
    assert evaluation_interval({'eval_every': 0}) == 0
    assert [e for e in range(1, 6) if should_evaluate(e, 5, 2)] == [2, 4, 5]
    try:
        evaluation_interval({'eval_every': -1})
    except ValueError:
        pass
    else:
        raise AssertionError('negative interval accepted')
    with tempfile.TemporaryDirectory() as root:
        p = dict(jepa_dir=root, epochs=2, save_every=3,
                 jepa_checkpoint=str(Path(root, 'checkpoint.pth.tar')),
                 jepa_model=str(Path(root, 'model.pth.tar')),
                 jepa_model_best=str(Path(root, 'model_best.pth.tar')))
        model = torch.nn.Linear(1, 1)
        model.anti_collapse = 'none'
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        trainer = Trainer(p, model, None, optimizer, scheduler, torch.device('cpu'), Mock())
        def train(loader, epoch):
            optimizer.zero_grad();model(torch.ones(1,1)).sum().backward();optimizer.step()
            return {'loss': 1., 'reconstruction loss': 1., 'latent_var': .2}
        trainer.train_one_epoch = train
        values = iter([5., 4., 3., .2, .4])
        def validate(loader):
            value = next(values)
            trainer.validation_metrics = {'loss': value, 'reconstruction loss': value/2, 'latent_var': .1}
            return value
        trainer.validate = validate
        evaluated = []
        def evaluate(epoch):
            evaluated.append(epoch)
            value = {2:.9, 4:.7, 5:.8}[epoch]
            return {mode: {'oracle': {'f1':value, 'threshold':.5}}
                    for mode in ('window','timeseries','non_overlap_window')}
        rolling = []
        original_update = Checkpoints.update
        def update(manager, epoch, *args, **kwargs):
            result = original_update(manager, epoch, *args, **kwargs)
            path = Path(root, 'checkpoint.pth.tar')
            saved_epoch = torch.load(path,weights_only=False)['epoch'] if path.exists() else None
            rolling.append((epoch,saved_epoch))
            assert not list(Path(root).glob('epoch_*.pth.tar'))
            return result
        with patch.object(Checkpoints, 'update', update):
            trainer.fit([], [], eval_every=2, eval_fn=evaluate)
        start, best = Trainer.resume(p, model, optimizer, scheduler, Mock())
        assert (start, best) == (2, 4.)
        p['epochs'] = 5
        with patch.object(Checkpoints, 'update', update):
            trainer.fit([], [], start_epoch=start, best_val_loss=best, eval_every=2, eval_fn=evaluate)
        assert evaluated == [2,4,5]
        assert rolling == [(1,None),(2,2),(3,3),(4,4),(5,5)]
        assert not Path(root,'last.pth.tar').exists()
        last = torch.load(Path(root,'checkpoint.pth.tar'),weights_only=False)
        final = torch.load(Path(root,'final.pth.tar'),weights_only=False)
        assert last['epoch'] == final['epoch'] == 5
        metadata = json.loads(Path(root, 'best_metrics.json').read_text())
        assert metadata['epoch'] == 5
        assert not list(Path(root).glob('*_epoch_*.pth.tar'))
        for path in Path(root).glob('*.pth.tar'):
            assert not path.is_symlink()
            assert metadata['checkpoints'][path.name]['epoch'] == torch.load(path, weights_only=False)['epoch']
        chosen = torch.load(Path(root, 'best_validation_loss.pth.tar'),weights_only=False)
        assert chosen['epoch'] == 4 and chosen['selection']['value'] == .2
        for mode in ('window','timeseries','non_overlap_window'):
            chosen = torch.load(Path(root, f'best_test_{mode}_oracle_f1.pth.tar'),weights_only=False)
            assert chosen['epoch'] == 2 and chosen['selection']['value'] == .9
            assert last['best'][f'{mode}_oracle_f1'] == .9
        assert Trainer.resume(p, model, optimizer, scheduler, Mock()) == (5,.2)
        # Selection metadata can be newer than an interrupted periodic file.
        recovered = Checkpoints(p, model, optimizer, scheduler,
                                best={'validation_loss':9., 'window_oracle_f1':.1})
        assert recovered.best['validation_loss'] == .2
        assert recovered.best['window_oracle_f1'] == .9
        assert recovered.saved_checkpoints == metadata['checkpoints']
    print('Cadence, periodic/evaluation/final snapshots, best loss/oracle and resume: OK')


if __name__ == '__main__':
    main()
