"""Warmed-up fp32 model scoring, excluding loading, transfers, metrics and I/O."""

import time

import numpy as np
import torch


@torch.inference_mode()
def benchmark_inference(model, series, window, batch_size, device, settings):
    warmup = int(settings.get("warmup_batches", 5))
    repeats = int(settings.get("measured_batches", 20))
    if warmup < 0 or repeats < 1:
        raise ValueError("benchmark needs nonnegative warmup_batches and positive measured_batches")
    count = min(batch_size, len(series) - window + 1)
    if count < 1:
        raise ValueError("benchmark series is shorter than the input window")
    starts = np.linspace(0, len(series) - window, count, dtype=int)
    windows = np.stack([series[start:start + window] for start in starts])
    batch = torch.from_numpy(windows).permute(0, 2, 1).contiguous().to(device).float()
    model.eval()
    for _ in range(warmup):
        model.score(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        model.score(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "cpu_threads": torch.get_num_threads(), "dtype": "float32", "batch_size": count,
        "warmup_batches": warmup, "measured_batches": repeats,
        "timed_seconds": elapsed, "batch_ms": elapsed * 1000 / repeats,
        "ms_per_window": elapsed * 1000 / (repeats * count),
        "windows_per_second": repeats * count / elapsed,
        "input_timesteps_per_second": repeats * count * window / elapsed,
        "scope": "model.score on device-resident batches; excludes transfers, metrics and I/O",
    }
