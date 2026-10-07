"""Regression checks for the Gaussian target of LeWM's SIGReg loss."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.sigreg import SIGReg  # noqa: E402


def main():
    torch.manual_seed(11)
    normal = torch.randn(1024, 12)
    sigreg = SIGReg(num_slices=64)

    def measure(samples):
        torch.manual_seed(7)
        return sigreg.statistic(samples).item()

    baseline = measure(normal)
    shifted = measure(normal + 5.0)
    narrowed = measure(normal * 0.01)
    assert shifted > baseline * 5, (baseline, shifted)
    assert narrowed > baseline * 5, (baseline, narrowed)

    # At zero, every projection is zero; the Epps--Pulley integral is known.
    collapsed = torch.zeros(32, 12)
    expected = 32 * ((1 - sigreg.phi).square() * sigreg.weights).sum()
    assert torch.allclose(sigreg.statistic(collapsed), expected, rtol=1e-5)

    # The production entry point must retain gradients through both streams.
    clean = normal[:32].reshape(32, 12, 1).requires_grad_()
    injected = (normal[32:64] + 2.0).reshape(32, 12, 1).requires_grad_()
    loss = sigreg({"clean": clean, "injected": injected})
    loss.backward()
    assert torch.isfinite(loss)
    assert clean.grad is not None and torch.isfinite(clean.grad).all()
    assert injected.grad is not None and torch.isfinite(injected.grad).all()
    assert clean.grad.abs().sum() > 0 and injected.grad.abs().sum() > 0
    print("SIGReg Gaussian-target checks OK")


if __name__ == "__main__":
    main()
