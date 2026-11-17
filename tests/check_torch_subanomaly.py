"""Run with ./venv/bin/python tests/check_torch_subanomaly.py."""

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.augment import SubAnomaly, TorchSubAnomaly


def check_legacy_parity(device):
    """Match every legacy branch using controlled random choices."""
    legacy = SubAnomaly(0.99)
    tensor_only = TorchSubAnomaly(0.99)
    for shape in ((32,), (32, 10)):
        source = torch.arange(1, 1 + int(np.prod(shape)), dtype=torch.float64,
                              device=device).reshape(shape)
        for anomaly_type in range(4 if len(shape) == 1 else 5):
            # Shared length/start, then channel count/index for multivariate.
            draws = [12, 3] + ([1, 2] if len(shape) == 2 else []) + [anomaly_type]
            if anomaly_type == 0:
                draws += [3]  # Frequency compression factor.
            original = source.clone()
            with patch.object(np.random, "randint", side_effect=draws), \
                    patch.object(np.random, "uniform", return_value=0.75), \
                    patch.object(np.random, "normal", return_value=1.25), \
                    patch.object(np.random, "rand", side_effect=lambda *s: np.full(s, 0.75)):
                expected = legacy(source.cpu().numpy())

            def constant_rand(size, **kwargs):
                return torch.full(size, 0.75, **kwargs)

            with patch.object(tensor_only, "_randint", side_effect=draws), \
                    patch.object(torch, "rand", side_effect=constant_rand), \
                    patch.object(torch, "randn", return_value=source.new_tensor(0.5)):
                actual = tensor_only(source)
            torch.testing.assert_close(actual, torch.as_tensor(expected, device=device))
            assert actual.device == source.device and actual.dtype == source.dtype
            assert actual.shape == source.shape
            torch.testing.assert_close(source, original)

    # Multivariate direct injection: vector scaling and clipped/end-extended spans.
    source = torch.arange(40, dtype=torch.float64, device=device).reshape(20, 2)
    for length, start, trend_end in ((7, 3, False), (30, 17, False), (4, 6, True)):
        kwargs = dict(subsequence_length=length, start_index=start,
                      compression_factor=3, trend_factor=0, trend_end=trend_end)
        expected = legacy.inject_frequency_anomaly(
            source.cpu().numpy(), scale_factor=np.array([0.5, 2.0]), **kwargs)
        actual = tensor_only.inject_frequency_anomaly(
            source, scale_factor=source.new_tensor([0.5, 2.0]), **kwargs)
        torch.testing.assert_close(actual, torch.as_tensor(expected, device=device))


def check_tensor_workflow(device):
    augment = TorchSubAnomaly(0.99)
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for shape in ((1,), (2,), (32,), (1, 1), (2, 2), (32, 3), (32, 16)):
            source = torch.linspace(1, 2, int(np.prod(shape)), device=device,
                                    dtype=dtype).reshape(shape)
            original = source.clone()
            # Fail immediately if tensor augmentation calls any NumPy RNG or
            # attempts to convert a tensor to NumPy or move it to the CPU.
            with patch.object(np.random, "randint", side_effect=AssertionError("NumPy RNG")), \
                    patch.object(np.random, "uniform", side_effect=AssertionError("NumPy RNG")), \
                    patch.object(np.random, "normal", side_effect=AssertionError("NumPy RNG")), \
                    patch.object(np.random, "rand", side_effect=AssertionError("NumPy RNG")), \
                    patch.object(torch.Tensor, "numpy", side_effect=AssertionError("NumPy conversion")), \
                    patch.object(torch.Tensor, "cpu", side_effect=AssertionError("CPU transfer")):
                torch.manual_seed(123)
                actual = augment(source)
                torch.manual_seed(123)
                repeated = augment(source)
                direct = augment.inject_frequency_anomaly(source.reshape(shape[0], -1))
            torch.testing.assert_close(actual, repeated)
            torch.testing.assert_close(source, original)
            assert actual.shape == shape
            assert actual.dtype == dtype and actual.device == source.device
            assert direct.dtype == dtype and direct.device == source.device
            assert bool(torch.isfinite(actual).all())

    # A transposed input exercises noncontiguous views and a real backward pass.
    source = torch.ones(12, 32, device=device, requires_grad=True).t()
    torch.manual_seed(5)
    actual = augment(source)
    actual.sum().backward()
    assert source._base.grad is not None
    assert bool(torch.isfinite(source._base.grad).all())

    # Only the requested span changes; shapelet noise is shared across features.
    source = torch.arange(40, dtype=torch.float32, device=device).reshape(20, 2)
    actual = augment.inject_frequency_anomaly(
        source, subsequence_length=7, start_index=3, compression_factor=1,
        scale_factor=1, trend_factor=0, shapelet_factor=True)
    torch.testing.assert_close(actual[:3], source[:3])
    torch.testing.assert_close(actual[10:], source[10:])
    noise = actual[3:10] - source[3]
    assert bool(((noise >= 0) & (noise < 0.1)).all())
    torch.testing.assert_close(noise[:, 0], noise[:, 1])


def check_validation():
    augment = TorchSubAnomaly(0.99)
    for source, exception in (([1.0], TypeError), (torch.ones(3, dtype=torch.int64), TypeError),
                              (torch.empty(0), ValueError), (torch.ones(2, 3, 4), ValueError)):
        try:
            augment(source)
        except exception:
            pass
        else:
            raise AssertionError(f"expected {exception.__name__}")
    for kwargs in (dict(subsequence_length=0), dict(compression_factor=0),
                   dict(start_index=-1), dict(start_index=20)):
        try:
            augment.inject_frequency_anomaly(torch.ones(20, 2), **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


if __name__ == "__main__":
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        check_legacy_parity(device)
        check_tensor_workflow(device)
        print(f"TorchSubAnomaly parity, tensor-only workflow, and gradients passed on {device}.")
    check_validation()
    print("TorchSubAnomaly input validation passed.")
