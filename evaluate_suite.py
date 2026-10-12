"""Run the complete evaluation suite for configured checkpoints and input cases."""

import argparse
import json

from utils.evaluation_suite import build_evaluation_plan, run_evaluation_suite


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="dataset-specific evaluation manifest")
    parser.add_argument("--config-env", default="configs/env.yml")
    parser.add_argument("--experiments", nargs="+", help="exact experiment keys")
    parser.add_argument("--device", help="override device for all evaluations")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    jobs, output, version = build_evaluation_plan(
        args.manifest, args.config_env, experiments=args.experiments, device=args.device)
    print(f"Evaluation suite: {len(jobs)} jobs -> {output}")
    if args.dry_run:
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
    rows = run_evaluation_suite(jobs, output, args.config_env, version, fail_fast=args.fail_fast)
    print(f"Completed {sum(row['status'] == 'completed' for row in rows)}/{len(rows)}; "
          f"summary: {output / 'summary.json'}")
    if any(row["status"] != "completed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
