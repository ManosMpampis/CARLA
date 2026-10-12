"""Evaluate explicit checkpoints with independent window/crop cases."""

import copy
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
import traceback

import yaml

from utils.config import DEFAULT_VERSION, load_experiment_config, merge_config, validate_experiment_name
from utils.experiment_suite import Experiment, build_plan, checkpoint_path, score_experiment, series_parts, _load_yaml


@dataclass
class EvaluationJob:
    experiment: Experiment
    checkpoint_name: str
    case_name: str
    output_dir: Path

    @property
    def key(self):
        return f"{self.experiment.key}/{self.checkpoint_name}/{self.case_name}"


def build_evaluation_plan(manifest_path, env_path, *, experiments=None, device=None):
    """Read-only planning; explicit checkpoint paths are relative to this manifest."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_yaml(manifest_path)
    name = validate_experiment_name(manifest.get("name", "default"), "evaluation name")
    version = manifest.get("version", DEFAULT_VERSION)
    source = manifest_path.parent / manifest["experiments_manifest"]
    entries = manifest.get("experiments")
    if not isinstance(entries, list) or not entries:
        raise ValueError("evaluation manifest needs a nonempty experiments list")
    selected = [entry for entry in entries if entry.get("enabled", True)]
    if experiments:
        missing = set(experiments) - {entry["experiment"] for entry in selected}
        if missing:
            raise ValueError(f"unknown evaluation experiments {sorted(missing)}")
        selected = [entry for entry in selected if entry["experiment"] in experiments]
    if not selected:
        raise ValueError("selection contains no evaluation experiments")
    plan, root = build_plan(source, env_path, version,
                            experiments=[entry["experiment"] for entry in selected])
    sources = {exp.key: exp for exp in plan}
    identities = {(exp.config["train_db_name"], exp.config["fname"]) for exp in plan}
    if len(identities) != 1:
        raise ValueError("an evaluation manifest must select one dataset and one series")
    dataset, _ = identities.pop()
    output_root = manifest.get("output_dir")
    output_root = ((manifest_path.parent / output_root).resolve() if output_root
                   else Path(root).resolve() / dataset / "evaluations" / name)
    if not manifest.get("output_dir"):
        output_root = output_root.joinpath(*series_parts(plan[0].config))
    jobs, keys = [], set()
    for entry in selected:
        exp = sources[entry["experiment"]]
        checkpoints = entry.get("checkpoints")
        cases = entry.get("evaluations", [{"name": "default"}])
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError(f"{exp.key}: checkpoints must be a nonempty list")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"{exp.key}: evaluations must be a nonempty list")
        for checkpoint in checkpoints:
            label = validate_experiment_name(checkpoint["name"], "checkpoint name")
            weights = ((manifest_path.parent / checkpoint["path"]).resolve()
                       if checkpoint.get("path") else checkpoint_path(exp, root, version).resolve())
            config_path = ((manifest_path.parent / checkpoint["config"]).resolve()
                           if checkpoint.get("config") else weights.parent / "resolved_config.yml")
            if checkpoint.get("config") or config_path.is_file():
                base = load_experiment_config(config_path)
            else:
                base = copy.deepcopy(exp.config)
                config_path = exp.config_path
            for case in cases:
                case = copy.deepcopy(case)
                case_name = validate_experiment_name(case.pop("name"), "evaluation case name")
                config = merge_config(base, manifest.get("common", {}))
                config = merge_config(config, entry.get("overrides", {}))
                config = merge_config(config, checkpoint.get("overrides", {}))
                config["evaluation"] = merge_config(config.get("evaluation", {}) or {}, case)
                if device is not None:
                    config["device"] = device
                config.update(stage="score", score_checkpoint=str(weights))
                job = EvaluationJob(replace(exp, config_path=config_path, config=config,
                                            dependency=None, run_version=None),
                                    label, case_name, output_root / exp.key / label / case_name)
                if job.key in keys:
                    raise ValueError(f"duplicate evaluation job {job.key}")
                keys.add(job.key)
                jobs.append(job)
    return jobs, output_root, version


def run_evaluation_suite(jobs, output_root, env_path, version, *, fail_fast=False):
    """Never train dependencies; persist every outcome and a partial suite summary."""
    rows = []
    output_root = Path(output_root)
    summary_path = output_root / "summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    combined = {row["job"]: row for row in existing}
    for job in jobs:
        started = time.monotonic()
        row = {"job": job.key, "checkpoint": job.experiment.config["score_checkpoint"],
               "output_dir": str(job.output_dir)}
        print(f"EVALUATE {job.key} -> {job.output_dir}", flush=True)
        try:
            if not Path(row["checkpoint"]).is_file():
                raise FileNotFoundError(f"configured checkpoint missing: {row['checkpoint']}")
            job.output_dir.mkdir(parents=True, exist_ok=True)
            (job.output_dir / "resolved_config.yml").write_text(
                yaml.safe_dump(job.experiment.config, sort_keys=False))
            report = score_experiment(job.experiment, env_path, version,
                                      output_dir=job.output_dir)
            row.update(status="completed", evaluation=report["evaluation"],
                       metrics_path=str(job.output_dir / "metrics.json"))
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            job.output_dir.mkdir(parents=True, exist_ok=True)
            (job.output_dir / "suite_error.txt").write_text(traceback.format_exc())
            print(f"FAILED {job.key}: {exc}", flush=True)
        row["seconds"] = round(time.monotonic() - started, 2)
        rows.append(row)
        combined[job.key] = row
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(list(combined.values()), indent=2))
        fields = ["job", "status", "checkpoint", "output_dir", "seconds", "error",
                  "timeseries_f1", "window_f1", "non_overlap_window_f1", "vus_pr", "vus_roc"]
        with (output_root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for saved in combined.values():
                flat = {key: saved[key] for key in fields if key in saved}
                evaluation = saved.get("evaluation", {})
                for procedure in ("timeseries", "window", "non_overlap_window"):
                    if procedure in evaluation:
                        flat[f"{procedure}_f1"] = evaluation[procedure]["calibrated"]["f1_no_pa"]
                if "timeseries" in evaluation:
                    for metric in ("vus_pr", "vus_roc"):
                        flat[metric] = evaluation["timeseries"][metric]
                writer.writerow(flat)
        if fail_fast and row["status"] != "completed":
            break
    return rows
