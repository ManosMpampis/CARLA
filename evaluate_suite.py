"""Run the complete evaluation suite for configured checkpoints and input cases."""

import argparse
import json
from pathlib import Path

import yaml

from utils.evaluation_suite import build_evaluation_plan, run_evaluation_suite
from utils.evaluation_summary import write_evaluation_overview
from utils.experiment_suite import load_suite_manifest


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="dataset-specific evaluation manifest")
    parser.add_argument("--config-env", default="configs/env.yml")
    parser.add_argument("--experiments", nargs="+", help="exact experiment keys")
    parser.add_argument("--device", help="override device for all evaluations")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    machines = parser.add_mutually_exclusive_group()
    machines.add_argument("--fname", help="one machine file")
    machines.add_argument("--machines", nargs="+", help="specific machine files")
    machines.add_argument("--all-machines", action="store_true", help="evaluate each SMD machine separately")
    parser.add_argument("--dataset-root")
    parser.add_argument("--skip-completed", action="store_true", help="resume the same evaluation suite")
    args = parser.parse_args()
    filenames = args.machines or [args.fname]
    if args.all_machines:
        from utils.mypath import MyPath

        manifest = Path(args.manifest).resolve()
        document = yaml.safe_load(manifest.read_text())
        common = load_suite_manifest(manifest.parent / document["experiments_manifest"]).get("common", {})
        if common.get("train_db_name") != "smd":
            parser.error("--all-machines requires an SMD manifest")
        data_root = Path(args.dataset_root or common.get("dataset_root") or MyPath.db_root_dir("smd"))
        filenames = sorted(path.name for path in (data_root / "train").glob("machine-*.txt"))
        if not filenames:
            parser.error(f"no SMD machines in {data_root / 'train'}")
    plans = [build_evaluation_plan(
        args.manifest, args.config_env, experiments=args.experiments, device=args.device,
        fname=fname, dataset_root=args.dataset_root) for fname in filenames]
    print(f"Evaluation suite: {sum(len(jobs) for jobs, _, _ in plans)} jobs across {len(plans)} series")
    if args.dry_run:
        for jobs, output, _ in plans:
            for job in jobs:
                print(f"{job.key}\n  weights: {job.experiment.config['score_checkpoint']}"
                      f"\n  evaluation: {json.dumps(job.experiment.config['evaluation'])}"
                      f"\n  output: {job.output_dir}")
        return
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            parser.error("--cpu-threads must be positive")
        import torch

        torch.set_num_threads(args.cpu_threads)
    failed = False
    for jobs, output, version in plans:
        rows = run_evaluation_suite(jobs, output, args.config_env, version,
                                    fail_fast=args.fail_fast, skip_completed=args.skip_completed)
        print(f"Completed {sum(row['status'] == 'completed' for row in rows)}/{len(rows)}; "
              f"summary: {output / 'summary.json'}", flush=True)
        if output.name.startswith("machine-"):
            write_evaluation_overview(output.parent)
        failed |= any(row["status"] != "completed" for row in rows)
        if failed and args.fail_fast:
            break
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
