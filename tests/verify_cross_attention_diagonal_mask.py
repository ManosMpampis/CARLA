"""Exact attention and model/checkpoint contracts for optional strict lower-triangular attention."""

import copy
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.lewm_cross_attention import CrossAttentionPredictor
from models.builders import get_cross_attention_model
from utils.config import load_experiment_config


def expect_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('expected ValueError')


def verify_attention():
    for q_len, k_len in ((4, 4), (3, 6), (6, 3), (1, 1), (3, 1)):
        for heads in (1, 2):
            for masked in (False, True):
                q = torch.randn(2, 4, q_len, requires_grad=True)
                k = torch.randn(2, 4, k_len, requires_grad=True)
                v = torch.randn(2, 6, k_len, requires_grad=True)
                predictor = CrossAttentionPredictor(torch.nn.Identity(), 4, 6, heads,
                                                     diagonal_mask=masked)
                def split(x):
                    return x.reshape(2, heads, x.size(1) // heads, x.size(2)).transpose(-1, -2)
                logits = split(q) @ split(k).transpose(-1, -2) / (4 // heads) ** 0.5
                if masked:
                    excluded = torch.arange(k_len)[None, :] >= torch.arange(q_len)[:, None]
                    logits = logits.masked_fill(excluded, -torch.inf)
                expected = (logits.softmax(-1).nan_to_num(0.0) @ split(v)).transpose(-1, -2).reshape(2, 6, q_len)
                actual = predictor({'query': q, 'key': k, 'value': v})
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                assert all(z.grad is not None and torch.isfinite(z.grad).all() for z in (q, k, v))
    # The first query has no allowed keys; the second sees only key 0.
    predictor = CrossAttentionPredictor(torch.nn.Identity(), 2, 2, diagonal_mask=True)
    qkv = {'query': torch.zeros(1, 2, 2), 'key': torch.zeros(1, 2, 2),
           'value': torch.tensor([[[1., 9.], [3., 7.]]])}
    expected = torch.tensor([[[0., 1.], [0., 3.]]])
    torch.testing.assert_close(predictor(qkv), expected)
    singleton = predictor({name: z[..., :1] for name, z in qkv.items()})
    assert singleton.count_nonzero() == 0
    # Later/current keys and values cannot affect an earlier attention output.
    qkv = {name: torch.randn(2, 2, 6, requires_grad=True)
           for name in ('query', 'key', 'value')}
    output = predictor(qkv)
    output[..., 3].sum().backward()
    for name in ('key', 'value'):
        assert qkv[name].grad[..., 3:].count_nonzero() == 0
        assert qkv[name].grad[..., :3].abs().sum() > 0
    changed = {name: z.detach().clone() for name, z in qkv.items()}
    for name in ('key', 'value'):
        changed[name][..., 3:] += 100
    torch.testing.assert_close(output[..., :4], predictor(changed)[..., :4], rtol=0, atol=0)
    for invalid in ('false', 1, None):
        expect_error(lambda: CrossAttentionPredictor(torch.nn.Identity(), diagonal_mask=invalid))


def verify_models():
    root = Path(__file__).resolve().parents[1]
    for arm in ('observed_features', 'observed_features_tcn', 'rec_ca'):
        p = load_experiment_config(root / 'configs/lewm_encoder/time/cross_attention' / f'{arm}.yml')
        assert p['cross_attention_kwargs']['diagonal_mask'] is False
        p['model_kwargs'] = {'in_channels': 4, 'enc_channels': [8, 12],
                             'enc_strides': [2, 2], 'norm': 'batch', 'dropout': 0.0}
        if arm == 'rec_ca':
            p['cross_attention_kwargs'].update(history_length=32, future_length=16)
        model = get_cross_attention_model(p).eval()
        legacy = copy.deepcopy(model.state_dict())
        legacy['_extra_state'].pop('diagonal_mask')
        legacy['_extra_state'].pop('attention_mask_layout')
        get_cross_attention_model(p).load_state_dict(legacy)
        p['cross_attention_kwargs']['diagonal_mask'] = True
        masked = get_cross_attention_model(p)
        expect_error(lambda: masked.load_state_dict(legacy))
        old_diagonal = copy.deepcopy(masked.state_dict())
        old_diagonal['_extra_state'].pop('attention_mask_layout')
        expect_error(lambda: masked.load_state_dict(old_diagonal))
        # Match weights while preserving each model's task metadata.
        weights = copy.deepcopy(model.state_dict())
        weights['_extra_state'] = masked.get_extra_state()
        masked.load_state_dict(weights)
        out = masked(torch.randn(3, 4, 48))
        assert out['recon'].shape == out['target'].shape
        out['recon'].square().mean().backward()
        query_weights = (masked.qkv_encoder.query.weight if arm == 'observed_features'
                         else masked.qkv_encoder.query.net[0].conv.weight)
        assert query_weights.grad is not None
        restored = get_cross_attention_model(p)
        restored.load_state_dict(masked.state_dict())
        assert restored.predictor.diagonal_mask is True
    print('Default inheritance, baseline/TCN/Rec+CA, gradients and checkpoint semantics: OK')


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.manual_seed(4)
    verify_attention()
    verify_models()
    print('verify_cross_attention_diagonal_mask: OK')
