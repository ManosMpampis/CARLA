"""Measure classification inference latency and memory usage.

Example:
    python benchmark_classification.py \
        --fname machine-1-1.txt \
        --version ./best_models/batch/my-experiment \
        --config_exp configs/classification/experiments/base.yml \
        --warmup 10 --iterations 100
"""

import argparse
import math
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import torch

from utils.common_config import get_model
from utils.config import create_config
from utils.utils import clean_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a saved CARLA classification model on synthetic input."
    )
    parser.add_argument("--config_env", default="configs/env.yml")
    parser.add_argument(
        "--config_exp", default="configs/classification/experiments/base.yml"
    )
    parser.add_argument("--fname", required=True, help="Dataset file name used by the config")
    parser.add_argument(
        "--version",
        default=None,
        help="Experiment version containing the checkpoint (for example ./best_models/batch/run)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit classification checkpoint path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device (default: auto)",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return torch.device(requested)


def resolve_checkpoint(config, explicit_path):
    if explicit_path:
        checkpoint_path = Path(explicit_path)
    else:
        model_path = Path(config["classification_model"])
        checkpoint_path = (
            model_path
            if model_path.is_file()
            else Path(config["classification_checkpoint"])
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Classification checkpoint not found at "
            f"{checkpoint_path}. Pass --checkpoint or use the matching --version."
        )
    return checkpoint_path


def load_classification_model(config, checkpoint_path, device):
    model = get_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = clean_checkpoint(state_dict)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def model_memory(model):
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    return parameter_bytes, buffer_bytes


def process_max_rss_mib():
    """Return the process high-water RSS in MiB on Linux and macOS."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024**2)
    return rss / 1024


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def benchmark(model, inputs, device, warmup, iterations):
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        elapsed = []
        output_shape = None
        for _ in range(iterations):
            start = time.perf_counter()
            output = model(inputs)
            synchronize(device)
            elapsed.append(time.perf_counter() - start)
            if output_shape is None:
                output_shape = (
                    tuple(output["output"].shape)
                    if isinstance(output, dict)
                    else tuple(output.shape)
                )
            del output

    return elapsed, output_shape


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("batch-size must be positive, warmup non-negative, and iterations positive")

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    config = create_config(
        args.config_env, args.config_exp, args.fname, version=args.version
    )
    checkpoint_path = resolve_checkpoint(config, args.checkpoint)
    model = load_classification_model(config, checkpoint_path, device)

    channels = (
        config["res_kwargs"]["in_channels"]
        if args.channels is None
        else args.channels
    )
    window_size = config["wsz"] if args.window_size is None else args.window_size
    if channels <= 0 or window_size <= 0:
        raise ValueError("channels and window-size must be positive")

    inputs = torch.randn(
        args.batch_size, channels, window_size, device=device, dtype=torch.float32
    )
    rss_before = process_max_rss_mib()
    elapsed, output_shape = benchmark(
        model, inputs, device, args.warmup, args.iterations
    )
    rss_after = process_max_rss_mib()

    parameter_bytes, buffer_bytes = model_memory(model)
    total_seconds = sum(elapsed)
    mean_ms = statistics.mean(elapsed) * 1000
    p50_ms = statistics.median(elapsed) * 1000
    p95_ms = percentile(elapsed, 0.95) * 1000
    throughput = args.batch_size * args.iterations / total_seconds

    print("Classification inference benchmark")
    print(f"checkpoint: {os.fspath(checkpoint_path)}")
    print(f"device: {device}")
    print(f"input shape: {(args.batch_size, channels, window_size)}")
    print(f"output shape: {output_shape}")
    print(f"iterations: {args.iterations} (warmup: {args.warmup})")
    print(f"latency mean: {mean_ms:.3f} ms/batch")
    print(f"latency p50: {p50_ms:.3f} ms/batch")
    print(f"latency p95: {p95_ms:.3f} ms/batch")
    print(f"throughput: {throughput:.2f} samples/s")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"model parameters: {parameter_bytes / 1024**2:.2f} MiB")
    print(f"model buffers: {buffer_bytes / 1024**2:.2f} MiB")

    if device.type == "cuda":
        allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        reserved = torch.cuda.max_memory_reserved(device) / 1024**2
        print(f"peak GPU allocated: {allocated:.2f} MiB")
        print(f"peak GPU reserved: {reserved:.2f} MiB")
    else:
        print(f"process max RSS before inference: {rss_before:.2f} MiB")
        print(f"process max RSS after inference: {rss_after:.2f} MiB")
        print(f"process RSS increase: {max(0.0, rss_after - rss_before):.2f} MiB")


if __name__ == "__main__":
    main()
