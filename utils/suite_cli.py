"""Shared command-line runner for explicitly selected dataset suites."""

import argparse
from pathlib import Path
from uuid import uuid4

from utils.config import DEFAULT_VERSION
from utils.experiment_suite import (build_plan, checkpoint_path, experiment_dir,
                                    load_suite_manifest, run_suite)


def suite_cli(default_manifest, dataset, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config-env", default="configs/env.yml")
    parser.add_argument("--manifest", default=str(default_manifest))
    runs = parser.add_mutually_exclusive_group()
    runs.add_argument("--version", help="select a named run; otherwise use stable experiment/tag paths")
    runs.add_argument("--fresh", action="store_true", help="start a separate run with a random name")
    parser.add_argument("--frameworks", nargs="+")
    parser.add_argument("--experiments", nargs="+", help="exact experiment keys")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--dataset-root", help="override dataset directory")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--score", "--score-only", dest="score_only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    if dataset == "smd":
        machines = parser.add_mutually_exclusive_group()
        machines.add_argument("--fname", help="one SMD machine file (default from manifest)")
        machines.add_argument("--machines", nargs="+", help="explicit SMD machine files")
        machines.add_argument("--all-machines", action="store_true", help="run all experiments separately on every machine")
    args = parser.parse_args()
    version = args.version or (f"run-{uuid4().hex[:8]}" if args.fresh else DEFAULT_VERSION)
    filenames = [dataset]
    if dataset == "smd":
        filenames = args.machines or [args.fname]
        common = load_suite_manifest(args.manifest).get("common", {})
        if args.all_machines or common.get("fname") == "all":
            from utils.mypath import MyPath
            data_root = args.dataset_root or common.get("dataset_root") or MyPath.db_root_dir(dataset)
            filenames = sorted(path.name for path in (Path(data_root) / "train").glob("machine-*.txt"))
            if not filenames:
                parser.error(f"no SMD machine files in {data_root}/train")
        if len(set(filenames)) != len(filenames):
            parser.error("machine files must be unique")
    plans = []
    for fname in filenames:
        plan, root = build_plan(args.manifest, args.config_env, version,
                               frameworks=args.frameworks, experiments=args.experiments,
                               epochs=args.epochs, device=args.device, auto_resume=False,
                               reuse_saved_sources=args.fresh, dataset=dataset,
                               fname=fname, dataset_root=args.dataset_root)
        plans.append((plan, root))
        print(f"{dataset.upper()} run {version}: {len(plan)} experiments ({fname or 'manifest series'})")
        if args.dry_run:
            for exp in plan:
                print(f"{exp.key}: {exp.config['epochs']} epochs -> {experiment_dir(exp, root, version)}")
                if exp.dependency:
                    print(f"  source: {exp.dependency} -> {exp.config['pretrained_from']}")
                print(f"  selected weights: {checkpoint_path(exp, root, version)}")
    if args.dry_run:
        return
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error("--cpu-threads must be positive")
        import torch

        torch.set_num_threads(args.cpu_threads)
    failed = False
    for plan, root in plans:
        rows = run_suite(plan, root, args.config_env, version,
                         score_only=args.score_only, fail_fast=args.fail_fast)
        failed |= any(row["status"] != "completed" for row in rows)
        if failed and args.fail_fast:
            break
    if failed:
        raise SystemExit(1)
