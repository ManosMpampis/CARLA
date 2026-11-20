"""Run with ./venv/bin/python tests/check_torch_masking_steered.py."""

from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.augment import TorchSubAnomaly
from utils.masking_steered import SubAnomalyMaskCollator, TorchSubAnomalyMaskCollator


def controlled_randint(draws, tensor=False):
    values = iter(draws)

    def sample(low, high, *args, **kwargs):
        value = next(values)
        assert low <= value < high, (low, value, high)
        return torch.tensor(value, device=kwargs["device"]) if tensor else value

    return sample


def check_legacy_parity(device):
    legacy = SubAnomalyMaskCollator()
    collator = TorchSubAnomalyMaskCollator()
    assert isinstance(collator.sub, TorchSubAnomaly)
    bcw = torch.arange(1, 513, dtype=torch.float32, device=device).reshape(2, 8, 32)
    for source in (bcw, bcw.transpose(1, 2).contiguous().transpose(1, 2)):
        original = source.clone()
        for anomaly_type in range(5):
            # Sample a span, then inject twice into the same channel. Frequency
            # injection additionally samples its compression factor each time.
            channel_draws = [3, anomaly_type] + ([3] if anomaly_type == 0 else [])
            draws = ([7, 23, 2] + channel_draws * 2) * 2
            with patch.object(np.random, "randint", side_effect=controlled_randint(draws)), \
                    patch.object(np.random, "uniform", return_value=0.75), \
                    patch.object(np.random, "normal", return_value=1.25), \
                    patch.object(np.random, "rand", side_effect=lambda *s: np.full(s, 0.75)):
                expected = legacy(99, 32, source)
            with patch.object(torch, "randint", side_effect=controlled_randint(draws, tensor=True)), \
                    patch.object(torch, "rand", side_effect=lambda size, **kw: torch.full(size, 0.75, **kw)), \
                    patch.object(torch, "randn", return_value=source.new_tensor(0.5)):
                actual = collator(source)
            for key in ("input", "X_inj"):
                torch.testing.assert_close(actual[key], expected[key].to(device))
            assert actual["X_inj"].shape == bcw.shape
            outside = ~actual["input"].unsqueeze(1).expand_as(bcw)
            assert torch.equal(actual["X_inj"][outside], bcw[outside])
            torch.testing.assert_close(source, original)

    # The last time step can be masked; span sampling has the same bounds.
    draws = [7, 25, 3, 0]
    with patch.object(np.random, "randint", side_effect=controlled_randint(draws)):
        expected = legacy(2, 32)
    with patch.object(torch, "randint", side_effect=controlled_randint(
            [7, 25, 1, 0, 3, 0, 1, 0], tensor=True)), \
            patch.object(collator, "_inject_channel", side_effect=lambda temp, *args: temp):
        actual = collator(torch.ones(2, 1, 32, device=device))
    assert set(actual) == {"input", "X_inj"}
    torch.testing.assert_close(actual["input"], expected["input"].to(device))
    assert actual["input"][0, -1]


def check_tensor_workflow(device):
    collator = TorchSubAnomalyMaskCollator()
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        for channels, window in ((1, 1), (1, 2), (3, 17), (12, 32), (12, 3), (4, 4)):
            bcw = torch.ones(2, channels, window, device=device, dtype=dtype)
            for source in (bcw, bcw.transpose(1, 2).contiguous().transpose(1, 2)):
                original = source.clone()
                with ExitStack() as stack:
                    for method in ("randint", "uniform", "normal", "rand"):
                        stack.enter_context(patch.object(np.random, method,
                                                        side_effect=AssertionError("NumPy RNG")))
                    for method in ("numpy", "cpu"):
                        stack.enter_context(patch.object(torch.Tensor, method,
                                                        side_effect=AssertionError("CPU/NumPy conversion")))
                    torch.manual_seed(123)
                    actual = collator(source)
                    torch.manual_seed(123)
                    repeated = collator(source)
                assert actual["input"].shape == (2, window)
                assert actual["input"].dtype == torch.bool
                assert actual["input"].device == source.device
                assert bool(actual["input"].any(dim=1).all())
                inj = actual["X_inj"]
                assert inj.shape == bcw.shape and inj.dtype == dtype
                assert inj.device == source.device and bool(torch.isfinite(inj).all())
                outside = ~actual["input"].unsqueeze(1).expand_as(inj)
                assert torch.equal(inj[outside], source[outside])
                torch.testing.assert_close(source, original)
                for key in actual:
                    torch.testing.assert_close(actual[key], repeated[key])

    # Check gradients when the same channel is injected twice through a view.
    base = torch.ones(2, 32, 8, device=device, requires_grad=True)
    source = base.transpose(1, 2)
    with patch.object(collator, "_sample_span", return_value=(7, 23)), \
            patch.object(collator, "_randint", side_effect=[2, 3, 2, 3, 3] * 2):
        actual = collator(source)
    actual["X_inj"].sum().backward()
    expected_grad = torch.ones_like(base)
    expected_grad[:, 23:30, 3] = 24  # Repeated scaling by 8 and then 3.
    torch.testing.assert_close(base.grad, expected_grad)

    full = TorchSubAnomalyMaskCollator(min_ratio=1, max_ratio=1)
    assert bool(full(torch.ones(2, 1, 1, device=device))["input"].all())
    assert bool(full(torch.ones(2, 3, 32, device=device))["input"].all())
    empty = collator(torch.empty(0, 3, 32, device=device))
    assert empty["input"].shape == (0, 32) and empty["X_inj"].shape == (0, 3, 32)


def check_validation():
    collator = TorchSubAnomalyMaskCollator()
    invalid = ((None, TypeError), ([1.0], TypeError),
               (torch.ones(2, 32), ValueError),
               (torch.ones(2, 3, 32, dtype=torch.int64), TypeError),
               (torch.empty(2, 3, 0), ValueError),
               (torch.empty(2, 0, 32), ValueError))
    for source, exception in invalid:
        try:
            collator(source)
        except exception:
            pass
        else:
            raise AssertionError(f"expected {exception.__name__}")
    for low, high in ((-0.1, 0.9), (0.9, 0.1), (0.1, 1.1)):
        try:
            TorchSubAnomalyMaskCollator(low, high)
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid ratios to raise ValueError")


if __name__ == "__main__":
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        check_legacy_parity(device)
        check_tensor_workflow(device)
        print(f"Torch collator legacy parity, masks, layouts, tensor-only execution, and gradients passed on {device}.")
    check_validation()
    print("Torch collator input validation passed.")
