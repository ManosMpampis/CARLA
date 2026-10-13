"""Train any native experiment on TSB-AD and export benchmark-aligned scores."""

import argparse
from pathlib import Path
from uuid import uuid4

import yaml

from data.tsb_ad import read_tsb_series
from utils.config import DEFAULT_VERSION, merge_config, validate_experiment_name
from utils.experiment_suite import checkpoint_path, experiment_dir
from utils.tsb_ad_benchmark import build_benchmark_plan, run_benchmark, select_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="extracted TSB-AD-U or TSB-AD-M directory")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--files", nargs="+", help="series filenames relative to dataset root")
    selection.add_argument("--file-list", help="official Eva/Tuning CSV with file_name column")
    parser.add_argument("--manifest", default="configs/tsb_ad/experiments.yml")
    parser.add_argument("--config-env", default="configs/env.yml")
    parser.add_argument("--output-root", help="benchmark outputs; defaults to root_dir/benchmarks/tsb_ad")
    versions = parser.add_mutually_exclusive_group()
    versions.add_argument("--version", default=None)
    versions.add_argument("--fresh", action="store_true")
    parser.add_argument("--frameworks", nargs="+")
    parser.add_argument("--experiments", nargs="+", help="exact existing suite experiment keys")
    parser.add_argument("--overrides", help="YAML mapping merged into all selected configurations")
    parser.add_argument("--epochs", type=int, help="total epoch target, including already completed epochs")
    parser.add_argument("--device")
    parser.add_argument("--window", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float, help="also overrides restored optimizer LR on resume")
    parser.add_argument("--initialize-from", help="weights-only warm start; requires one explicit experiment and file")
    parser.add_argument("--window-policy", choices=("shrink", "strict"), default="shrink")
    parser.add_argument("--metrics", choices=("deferred", "tsb-ad"), default="deferred")
    parser.add_argument("--tsb-ad-root", help="optional official TSB-AD source checkout for metrics")
    parser.add_argument("--sliding-window", type=int, help="VUS metric buffer; defaults to upstream periodicity estimate")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    version = args.version or (f"run-{uuid4().hex[:8]}" if args.fresh else DEFAULT_VERSION)
    validate_experiment_name(version, "version")
    with open(args.config_env) as stream:
        env = yaml.safe_load(stream)
    root = Path(args.output_root or Path(env["root_dir"]) / "benchmarks" / "tsb_ad").resolve()
    overrides = {}
    if args.overrides:
        with open(args.overrides) as stream:
            overrides = yaml.safe_load(stream)
        if not isinstance(overrides, dict):
            parser.error("--overrides requires a YAML mapping")
    for arg, key in (("window", "wsz"), ("stride", "stride"),
                     ("val_fraction", "val_fraction"), ("batch_size", "batch_size")):
        value = getattr(args, arg)
        if value is not None:
            overrides[key] = value
    if args.learning_rate is not None:
        import math
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("--learning-rate must be finite and positive")
        overrides = merge_config(overrides, {"optimizer_kwargs": {"lr": args.learning_rate},
                                             "resume_learning_rate": args.learning_rate})
    try:
        files = select_files(args.dataset_root, files=args.files, file_list=args.file_list)
        if args.initialize_from and (not args.experiments or len(args.experiments) != 1 or len(files) != 1):
            parser.error("--initialize-from requires one --experiments key and one selected file")
        if args.metrics == "tsb-ad" and not args.dry_run:
            from utils.tsb_ad_metrics import load_metrics
            load_metrics(args.tsb_ad_root)  # fail before starting expensive training
        if args.sliding_window is not None and args.sliding_window < 1:
            parser.error("--sliding-window must be positive")
        if args.initialize_from and not Path(args.initialize_from).is_file():
            parser.error(f"initial weights missing: {args.initialize_from}")

        def plans():
            # Load one time series at a time: a full suite must not retain the
            # complete U/M collection in memory while fitting its first model.
            for file in files:
                try:
                    series = read_tsb_series(file)
                    plan = build_benchmark_plan(args.manifest, args.config_env, version, series, root,
                                                overrides=overrides, frameworks=args.frameworks,
                                                experiments=args.experiments, epochs=args.epochs,
                                                device=args.device, window_policy=args.window_policy)
                    if args.initialize_from:
                        target = next(exp for exp in plan if exp.key == args.experiments[0])
                        target.config["initialize_from"] = str(Path(args.initialize_from).resolve())
                        target.dependency = None  # complete weights already contain the encoder
                        plan = [target]
                    print(f"{file.name}: train prefix={series.train_end}, channels={series.channels}")
                    for exp in plan:
                        if exp.config.get("benchmark_planning_error"):
                            print(f"  INCOMPATIBLE {exp.key}: {exp.config['benchmark_planning_error']}")
                        print(f"  {exp.key}: window={exp.config['wsz']}, batch={exp.config['batch_size']} "
                              f"-> {experiment_dir(exp, root, version)}")
                        if exp.dependency:
                            print(f"    source: {exp.dependency} -> {exp.config['pretrained_from']}")
                        print(f"    validation-selected weights: {checkpoint_path(exp, root, version)}")
                    yield series, plan
                except (ValueError, FileNotFoundError) as exc:
                    print(f"FAILED planning {file.name}: {exc}", flush=True)
                    yield None, {"file": file.name, "experiment": "planning", "status": "failed",
                                 "error": f"{type(exc).__name__}: {exc}"}
        if args.dry_run:
            failed = False
            for series, _ in plans():
                failed |= series is None or any(exp.config.get("benchmark_planning_error") for exp in _)
                if failed and args.fail_fast:
                    break
            if failed:
                raise SystemExit(1)
            return
        if args.cpu_threads is not None:
            if args.cpu_threads < 1:
                parser.error("--cpu-threads must be positive")
            import torch
            torch.set_num_threads(args.cpu_threads)
        rows = run_benchmark(plans(), args.config_env, root, version, score_only=args.score_only,
                             fail_fast=args.fail_fast, metrics=args.metrics, tsb_ad_root=args.tsb_ad_root,
                             sliding_window=args.sliding_window, trace=not args.no_trace)
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Benchmark summary: {root / 'benchmark_runs' / version / 'summary.json'}")
    if any(row["status"] != "completed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
