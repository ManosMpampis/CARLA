"""Exercise the actual AE/VAE and shared Trainer epoch loops without training."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from data.jepa_dataset import JEPADataset
from utils import reconstruction_baselines as baselines
from utils.trainer import Trainer
from utils.evaluation_schedule import evaluation_interval


def baseline_steps(arm, interval):
    with tempfile.TemporaryDirectory() as root:
        config = {"arm": arm, "device": "cpu", "wsz": 8, "stride": 1,
                  "epochs": 5, "batch_size": 2, "jepa_dir": root,
                  "eval_every": interval, "model_kwargs": {}}
        class Dataset(JEPADataset):
            def __init__(self, *args, **kwargs):
                self._corpus = None
                self.series = []
                self.targets = []

        dataset = Dataset()
        model = Mock(total_stride=1)
        model.to.return_value = model
        calls = []
        def run_epoch(*args, **kwargs):
            return {"loss": 10.0 - args[6]}
        def evaluate(*args):
            calls.append(len(epochs))
            return {}
        epochs = []
        def epoch(*args, **kwargs):
            if kwargs["training"]:
                epochs.append(args[6])
            return run_epoch(*args, **kwargs)
        with patch.multiple(baselines, create_config=Mock(return_value=config),
                            get_ae_model=Mock(return_value=model),
                            get_vae_model=Mock(return_value=model),
                            get_jepa_datasets=Mock(return_value=(dataset, dataset)),
                            JEPADataset=Dataset,
                            get_train_dataloader=Mock(return_value=[]),
                            get_val_dataloader=Mock(return_value=[]),
                            get_optimizer=Mock(), get_scheduler=Mock(),
                            SummaryWriter=Mock(), _save=Mock(),
                            _run_epoch=epoch, evaluate=evaluate), \
             patch.object(torch, 'save'):
            baselines.train_arm(arm, SimpleNamespace(config_env='', config_exp='',
                                                    fname='', version=''))
        rows = [json.loads(line) for line in Path(root, 'epochs.jsonl').read_text().splitlines()]
        assert [row['epoch'] for row in rows] == [1, 2, 3, 4, 5]
        history_path = Path(root, 'test_evaluations.jsonl')
        history = [json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
        assert [row['epoch'] for row in history] == calls
        return calls


def trainer_steps(interval):
    trainer = object.__new__(Trainer)
    trainer.p = {"epochs": 5, "jepa_model": "unused", "jepa_model_best": "unused_best"}
    trainer.logger = Mock()
    trainer.optimizer = SimpleNamespace(param_groups=[{'lr': 0.1}])
    trainer.scheduler = Mock()
    trainer.model = Mock()
    trainer.train_one_epoch = Mock(return_value={"loss": 1.0})
    trainer.validate = Mock(side_effect=[5., 4., 3., 2., 1.])
    trainer.save_checkpoint = Mock()
    steps = []
    with patch.object(torch, 'save'):
        trainer.fit([], [], eval_every=interval,
                    eval_fn=lambda step: (steps.append(step) or 0.0))
    assert trainer.validate.call_count == 5
    validation_steps = [call.args[3] for call in trainer.logger.scalar_summary.call_args_list
                        if call.args[0] == "validation"]
    assert validation_steps == [1, 2, 3, 4, 5]
    return steps


def main():
    assert evaluation_interval({}) == 10
    assert evaluation_interval({"stage_c": {"eval_every": 0}}) == 0
    assert evaluation_interval({"eval_every": 2, "stage_c": {"eval_every": 0}}) == 2
    try:
        evaluation_interval({"eval_every": -1})
    except ValueError:
        pass
    else:
        raise AssertionError("negative cadence accepted")
    for interval, expected in ((1, [1, 2, 3, 4, 5]), (2, [2, 4, 5]), (0, [])):
        for arm in ('ae', 'vae'):
            actual = baseline_steps(arm, interval)
            assert actual == expected, (arm, interval, actual, expected)
        assert trainer_steps(interval) == expected
    print('Validation every epoch; test evaluation cadence and disable: OK')


if __name__ == '__main__':
    main()
