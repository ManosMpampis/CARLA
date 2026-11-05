"""Plain assertions for joint Rec/CA semantics and native training handoff."""

import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.builders import get_cross_attention_model, get_lewm_model
from models.lewm_cross_attention_tcn import CausalQueryTCN
from lewm_cross_attention import main as run_stage
from utils.common_config import get_criterion
from utils.config import experiment_base_dir, load_experiment_config


def config():
    return {
        "model_kwargs": {"in_channels": 4, "enc_channels": [8, 12],
                         "enc_strides": [2, 2], "norm": "batch", "dropout": 0.0},
        "cross_attention_model": "lewm_rec_ca",
        "cross_attention_kwargs": {"history_length": 32, "future_length": 16,
                                   "qk_channels": 4, "value_channels": 6,
                                   "num_heads": 2, "query_tcn_dropout": 0.0,
                                   "dropout": 0.0},
        "criterion": "lewm_rec_ca",
        "criterion_kwargs": {"sigreg_kwargs": {"num_slices": 4}},
    }


def verify_model():
    p = config()
    model = get_cross_attention_model(p).train()
    assert not model.encoder.training
    assert isinstance(model.qkv_encoder.query, CausalQueryTCN)
    assert isinstance(model.qkv_encoder.value, torch.nn.Conv1d)
    x = torch.randn(3, 4, 48)
    out = model(x, mask={"X_inj": x + 0.2 * torch.randn_like(x)})
    assert out["rec_full"].shape == (3, 12, 12)
    assert out["rec_future"].shape == out["recon"].shape == (3, 12, 4)
    assert out["qkv"]["value"].shape == (3, 6, 8)
    criterion = get_criterion(p)
    loss = criterion(out)
    expected = (loss["rec_loss"] + loss["ca_pred_loss"] + loss["similarity_loss"]
                + 0.1 * (loss["sigreg"] + loss["sigreg_tgt"]))
    torch.testing.assert_close(loss["loss"], expected)
    agreement_grads = torch.autograd.grad(loss["similarity_loss"],
                                         (out["rec_full"], out["recon"]), retain_graph=True)
    assert all(g.abs().sum() > 0 for g in agreement_grads)
    assert agreement_grads[0][..., :8].count_nonzero() == 0
    loss["loss"].backward()
    for branch in (model.rec_head, model.predictor, model.qkv_encoder):
        grads = [p.grad for p in branch.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert all(p.grad is None for p in model.encoder.parameters())
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    model.eval()
    before = model(x)
    changed = x.clone()
    changed[..., 32:] += 5 * torch.randn_like(changed[..., 32:])
    after = model(changed)
    for name in ('key', 'value'):
        torch.testing.assert_close(before["qkv"][name], after["qkv"][name], rtol=0, atol=0)
    assert not torch.equal(before["qkv"]["query"], after["qkv"]["query"])
    # Query TCN itself is causal and preserves token length.
    z = torch.randn(2, 12, 16)
    changed_z = z.clone()
    changed_z[..., 8:] += 10
    query = model.qkv_encoder.query
    assert query(z).shape == (2, 4, 16)
    torch.testing.assert_close(query(z)[..., :8], query(changed_z)[..., :8], rtol=0, atol=0)
    model.train()
    scores = model.score(x)
    assert model.training and not model.encoder.training
    assert scores["fused"].shape == (3, 16)
    torch.testing.assert_close(scores["fused"], (scores["signals"]["rec"] + scores["signals"]["ca"]) / 2)
    saved = copy.deepcopy(model.state_dict())
    restored = get_cross_attention_model(p).eval()
    restored.load_state_dict(saved)
    torch.testing.assert_close(before["recon"], restored(x)["recon"])
    altered = copy.deepcopy(p)
    altered["cross_attention_kwargs"].update(history_length=16, future_length=32)
    try:
        get_cross_attention_model(altered).load_state_dict(saved)
    except ValueError:
        pass
    else:
        raise AssertionError("changed geometry must reject checkpoint")
    paths = sorted((REPO / 'configs/lewm_encoder').glob('*/cross_attention/rec_ca.yml'))
    assert len(paths) == 6
    for path in paths:
        cfg = load_experiment_config(path)
        arm = get_cross_attention_model(cfg).eval()
        assert arm.crop_ranges(384) == ((0, 32), (32, 48))
        with torch.no_grad():
            full = arm(torch.randn(2, 38, 384))
        assert full["rec_full"].shape == (2, 128, 48)
        assert full["rec_future"].shape == full["recon"].shape == (2, 128, 16)
    print('Shapes, branch gradients, query causality, frozen encoder, configs and checkpoint task: OK')


def verify_runner():
    with tempfile.TemporaryDirectory(prefix='lewm-rec-ca-') as tmp:
        directory = Path(tmp)
        env = directory / 'env.yml'
        env.write_text(yaml.safe_dump({'root_dir': str(directory / 'results')}))
        source = directory / 'encoder.pth.tar'
        torch.save(get_lewm_model(config()).state_dict(), source)
        path = REPO / 'configs/lewm_encoder/time/cross_attention/rec_ca.yml'
        # Overrides replace entire nested dictionaries, avoiding inherited widths.
        patch = {**config(), 'train_db_name': 'synthetic', 'val_db_name': 'synthetic',
                 'synthetic_kwargs': {'n_steps': 256, 'test_n_steps': 160, 'n_channels': 4},
                 'val_fraction': 0.3, 'wsz': 48, 'stride': 16, 'batch_size': 2,
                 'epochs': 1, 'device': 'cpu', 'num_workers': 0, 'amp': False,
                 'eval_every': 0, 'pretrained_from': str(source),
                 'stage_b': {'masking': {'mode': 'none'}}, 'score_batch_size': 8,
                 'eval_window_size': 8}
        args = SimpleNamespace(config_env=str(env), config_exp=str(path), fname='toy', version='smoke')
        run_stage(args, patch)
        p = {**load_experiment_config(path), **patch}
        output = Path(experiment_base_dir(directory / 'results', p, 'toy', 'smoke')) / 'jepa_rec_ca'
        assert (output / 'model.pth.tar').is_file()
        state = torch.load(output / 'checkpoint.pth.tar', weights_only=False)
        assert state['next_epoch'] == 1
        run_stage(args, patch)
        run_stage(args, {**patch, 'stage': 'score'})
        for name in ('calibration.json', 'metrics.json'):
            assert (output / name).is_file()
    print('Synthetic train -> resume -> calibration and scoring: OK')


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(4)
    verify_model()
    verify_runner()
    print('verify_lewm_rec_ca: OK')
