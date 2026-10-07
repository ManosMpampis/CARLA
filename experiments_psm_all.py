"""Run every configured PSM framework and named variant; print/save results."""

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from utils.experiment_suite import build_plan, checkpoint_path, experiment_dir, run_suite


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-env", default="configs/env.yml")
    parser.add_argument("--manifest", default=str(Path(__file__).resolve().parent / "configs/psm/experiments.yml"))
    parser.add_argument("--version", help="run name; reuse to resume, change for a fresh run")
    parser.add_argument("--frameworks", nargs="+", help="ae vae lewm_encoder reconstruction cross_attention")
    parser.add_argument("--experiments", nargs="+", help="exact keys, e.g. lewm_encoder/time/cross_attention/context_input")
    parser.add_argument("--epochs", type=int, help="override training epochs for every selected experiment")
    parser.add_argument("--device", help="override device, e.g. cuda or cpu")
    parser.add_argument("--cpu-threads", type=int, help="optional torch CPU thread count")
    parser.add_argument("--score", "--score-only", dest="score_only", action="store_true", help="score existing validation-selected checkpoints")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failure")
    parser.add_argument("--dry-run", action="store_true", help="show the plan and paths without writing or training")
    args = parser.parse_args()
    version = args.version or datetime.now(ZoneInfo("Europe/Athens")).strftime("%Y-%m-%d-%H-%M-%S")
    plan, root = build_plan(args.manifest, args.config_env, version,
                            frameworks=args.frameworks, experiments=args.experiments,
                            epochs=args.epochs, device=args.device)
    print(f"PSM run {version}: {len(plan)} experiments (including dependencies)")
    if args.dry_run:
        for exp in plan:
            print(f"{exp.key}: {exp.config['epochs']} epochs -> {experiment_dir(exp, root, version)}")
            if exp.dependency:
                print(f"  source: {exp.dependency} -> {exp.config['pretrained_from']}")
            print(f"  selected weights: {checkpoint_path(exp, root, version)}")
        return
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error("--cpu-threads must be positive")
        import torch

        torch.set_num_threads(args.cpu_threads)
    rows = run_suite(plan, root, args.config_env, version,
                      score_only=args.score_only, fail_fast=args.fail_fast)
    if any(row["status"] != "completed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
