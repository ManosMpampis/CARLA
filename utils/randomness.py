"""Save training RNG state and isolate deterministic validation/preflight work."""

from contextlib import contextmanager
import random

import numpy as np
import torch


def capture_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def capture_module_rng(module):
    return {name: {str(device): generator.get_state() for device, generator in child._generators.items()}
            for name, child in module.named_modules() if hasattr(child, "_generators")}


def restore_module_rng(module, state):
    for name, child in module.named_modules():
        if not hasattr(child, "_generators"):
            continue
        child._generators = {}
        for device, value in (state or {}).get(name, {}).items():
            if device.startswith("cuda") and not torch.cuda.is_available():
                continue
            child._generators[torch.device(device)] = torch.Generator(device=device).set_state(value.cpu())


@contextmanager
def isolated_module_rng(module):
    """Use fresh seeded local generators without advancing the training copies."""
    saved = [(child, child._generators) for child in module.modules() if hasattr(child, "_generators")]
    try:
        for child, _ in saved:
            child._generators = {}
        yield
    finally:
        for child, generators in saved:
            child._generators = generators


@contextmanager
def isolated_rng(seed=None):
    state = capture_rng_state()
    try:
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        restore_rng_state(state)
