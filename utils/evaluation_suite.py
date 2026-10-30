"""Evaluate explicit checkpoints with independent window/crop cases."""

import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time
import traceback

import yaml

from utils.config import DEFAULT_VERSION, dataset_output_dir, load_experiment_config, merge_config, validate_experiment_name
from utils.experiment_suite import Experiment, build_plan, checkpoint_path, score_experiment, series_parts, _load_yaml
from utils.evaluation_summary import write_mode_csvs


@dataclass
class EvaluationJob:
    experiment: Experiment
    checkpoint_name: str
    case_name: str
    output_dir: Path

    @property
    def key(self):
        return f"{self.experiment.key}/{self.checkpoint_name}/{self.case_name}"


def build_evaluation_plan(manifest_path, env_path, *, experiments=None, device=None,
                          fname=None, dataset_root=None):
    """Read-only planning; explicit checkpoint paths are relative to this manifest."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_yaml(manifest_path)
    name = validate_experiment_name(manifest.get("name", "default"), "evaluation name")
    version = manifest.get("version", DEFAULT_VERSION)
    source = manifest_path.parent / manifest["experiments_manifest"]
    entries = manifest.get("experiments")
    if entries == "all":
        available, _ = build_plan(source, env_path, version, fname=fname,
                                   dataset_root=dataset_root)
        entries = []
        for exp in available:
            cases = copy.deepcopy(manifest.get("evaluations", [{"name": "default"}]))
            if exp.runner == "cross_attention":
                for case in cases:
                    if not {"context_length", "target_length", "context_crop", "target_crop"} & case.keys():
                        case.update(context_crop=None, target_crop=None)
                cases.extend(copy.deepcopy(manifest.get("cross_attention_evaluations", [])))
            entries.append(dict(experiment=exp.key, checkpoints=[dict(name="selected")],
                                evaluations=cases))
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
                            experiments=[entry["experiment"] for entry in selected],
                            fname=fname, dataset_root=dataset_root)
    sources = {exp.key: exp for exp in plan}
    identities = {(exp.config["train_db_name"], exp.config["fname"],
                   exp.config.get("experiment_version")) for exp in plan}
    if len(identities) != 1:
        raise ValueError("an evaluation manifest must select one dataset, series and experiment_version")
    dataset, _, _ = identities.pop()
    output_root = manifest.get("output_dir")
    output_root = ((manifest_path.parent / output_root).resolve() if output_root
                   else Path(dataset_output_dir(root, plan[0].config)).resolve() / "evaluations" / name)
    if not manifest.get("output_dir"):
        output_root = output_root.joinpath(*series_parts(plan[0].config))
    elif fname is not None and fname != dataset:
        output_root /= fname
    jobs, keys = [], set()
    for entry in selected:
        exp = sources[entry["experiment"]]
        checkpoints = entry.get("checkpoints")
        cases = entry.get("evaluations", manifest.get("evaluations", [{"name": "default"}]))
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
                if {"context_length", "target_length"} & case.keys():
                    for key in ("context_crop", "target_crop"):
                        if key not in case:
                            config["evaluation"].pop(key, None)
                if device is not None:
                    config["device"] = device
                config.update(stage="score", score_checkpoint=str(weights))
                config["fname"] = exp.config["fname"]
                if dataset_root is not None:
                    config["dataset_root"] = str(dataset_root)
                job = EvaluationJob(replace(exp, config_path=config_path, config=config,
                                            dependency=None, run_version=None),
                                    label, case_name, output_root / exp.key / label / case_name)
                if job.key in keys:
                    raise ValueError(f"duplicate evaluation job {job.key}")
                keys.add(job.key)
                jobs.append(job)
    return jobs, output_root, version


def run_evaluation_suite(jobs, output_root, env_path, version, *, fail_fast=False,
                         skip_completed=False):
    """Never train dependencies; persist every outcome and a partial suite summary."""
    rows = []
    output_root = Path(output_root)
    summary_path = output_root / "summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    combined = {row["job"]: row for row in existing}
    for job in jobs:
        saved = combined.get(job.key)
        if (skip_completed and saved and saved.get("status") == "completed"
                and (job.output_dir / "metrics.json").is_file()
                and all((job.output_dir / f"{section}_metrics.csv").is_file()
                        for section in ("Test_window", "Test_non_overlap", "Test_timeseries"))):
            print(f"SKIP {job.key} (completed)", flush=True)
            rows.append(saved)
            continue
        started = time.monotonic()
        row = {"job": job.key, "checkpoint": job.experiment.config["score_checkpoint"],
               "output_dir": str(job.output_dir), "experiment_key": job.experiment.key,
               "checkpoint_name": job.checkpoint_name, "case": job.case_name,
               "fname": job.experiment.config["fname"]}
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
                       evaluation_settings=report["evaluation_settings"],
                       inference_speed=report.get("inference_speed", {}),
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
        write_mode_csvs(output_root, list(combined.values()))
        speed = row.get("inference_speed", {}).get("windows_per_second")
        timing = f", {speed:.1f} windows/s" if speed is not None else ""
        print(f"{row['status'].upper()} {job.key}: {row['seconds']:.2f}s{timing}", flush=True)
        if fail_fast and row["status"] != "completed":
            break
    return rows
