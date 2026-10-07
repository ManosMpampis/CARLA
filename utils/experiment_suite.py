"""Named experiment planning, native training, and comparable PSM reporting."""

import copy
import csv
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import yaml

from utils.config import (create_config, experiment_base_dir,
                          load_experiment_config, merge_config,
                          validate_experiment_name)


@dataclass
class Experiment:
    framework: str
    name: str
    runner: str
    config_path: Path
    config: dict
    dependency: str | None = None

    @property
    def key(self):
        if self.config.get("phase1_experiment") and self.runner in ("recon", "cross_attention"):
            branch = "reconstruction" if self.runner == "recon" else "cross_attention"
            return f"lewm_encoder/{self.config['phase1_experiment']}/{branch}/{self.name}"
        return f"{self.framework}/{self.name}"


# Existing training entries remain the authority for their training behavior.
RUNNERS = {
    "ae": ("utils.reconstruction_baselines:train_arm", "models.builders:get_ae_model", "best_validation_loss.pth.tar"),
    "vae": ("utils.reconstruction_baselines:train_arm", "models.builders:get_vae_model", "best_validation_loss.pth.tar"),
    "lewm": ("lewm:main", "models.builders:get_lewm_model", "model.pth.tar"),
    "recon": ("lewm_reconstruction:main", "models.builders:get_recon_model", "model.pth.tar"),
    "cross_attention": ("lewm_cross_attention:main", "models.builders:get_cross_attention_model", "model.pth.tar"),
}


def _load_yaml(path):
    with open(path) as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def experiment_dir(exp, root_dir, version):
    base = Path(experiment_base_dir(root_dir, exp.config, "psm", version))
    tag = exp.config.get("tag_jepa")
    return base / (f"jepa_{tag}" if tag else "jepa")


def checkpoint_path(exp, root_dir, version):
    return experiment_dir(exp, root_dir, version) / RUNNERS[exp.runner][2]


def build_plan(manifest_path, env_path, version, *, frameworks=None,
               experiments=None, epochs=None, device=None):
    """Resolve a manifest without writing directories; order source LEWM first."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _load_yaml(manifest_path)
    root_dir = _load_yaml(env_path)["root_dir"]
    validate_experiment_name(version, "version")
    entries = manifest.get("experiments", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest needs a nonempty experiments list")
    common = manifest.get("common", {})
    all_experiments = {}
    for entry in entries:
        if entry.get("enabled", True) is False:
            continue
        framework = validate_experiment_name(entry["framework"], "framework")
        name = validate_experiment_name(entry.get("experiment_name", "default"), "experiment_name")
        runner = entry["runner"]
        if runner not in RUNNERS:
            raise ValueError(f"unknown runner {runner}; expected {sorted(RUNNERS)}")
        config_path = (manifest_path.parent / entry["config"]).resolve()
        config = merge_config(load_experiment_config(config_path), common)
        config = merge_config(config, entry.get("overrides", {}))
        description = config.get("experiment_name")
        config.update(framework=framework, experiment_name=name, fname="psm",
                      train_db_name="psm", val_db_name="psm")
        # A named phase-one variant creates a new encoder parent. Descendants
        # follow their source's identity even when reusing a base config file.
        if runner == "lewm" and framework == "lewm_encoder":
            config["phase1_experiment"] = name
        dependency = entry.get("pretrained_experiment")
        if dependency and dependency.startswith("lewm_encoder/"):
            source_name = dependency.removeprefix("lewm_encoder/")
            config["phase1_experiment"] = validate_experiment_name(source_name, "pretrained_experiment")
        if description:
            config["experiment_description"] = description
        if epochs is not None:
            if epochs < 1:
                raise ValueError("epochs must be positive")
            config["epochs"] = epochs
        if device is not None:
            config["device"] = device
        if runner in ("ae", "vae"):
            config["arm"] = runner
        else:
            config["stage"] = {"lewm": "pretrain", "recon": "recon",
                               "cross_attention": "phase2"}[runner]
        if config["setup"] != "jepa":
            raise ValueError("suite entries must use setup=jepa")
        if int(config["model_kwargs"]["in_channels"]) != 25:
            raise ValueError(f"{framework}/{name}: PSM requires 25 input channels")
        exp = Experiment(framework, name, runner, config_path, config,
                         dependency)
        if exp.key in all_experiments:
            raise ValueError(f"duplicate experiment {exp.key}")
        all_experiments[exp.key] = exp

    if frameworks:
        missing = set(frameworks) - {exp.framework for exp in all_experiments.values()}
        if missing:
            raise ValueError(f"unknown frameworks {sorted(missing)}")
    if experiments:
        missing = set(experiments) - set(all_experiments)
        if missing:
            raise ValueError(f"unknown experiments {sorted(missing)}")
    selected = [key for key, exp in all_experiments.items()
                if (not frameworks or exp.framework in frameworks)
                and (not experiments or key in experiments)]
    if not selected:
        raise ValueError("selection contains no experiments")
    plan, visited, visiting = [], set(), set()

    def visit(key):
        if key in visiting:
            raise ValueError(f"cyclic pretrained_experiment dependency at {key}")
        if key in visited:
            return
        if key not in all_experiments:
            raise ValueError(f"missing pretrained_experiment {key}")
        exp = all_experiments[key]
        visiting.add(key)
        if exp.dependency:
            if exp.runner not in ("recon", "cross_attention"):
                raise ValueError(f"{key}: only phase-two runners accept pretrained_experiment")
            visit(exp.dependency)
            source = all_experiments[exp.dependency]
            if source.runner != "lewm":
                raise ValueError(f"{key}: pretrained_experiment must use the lewm runner")
            if exp.config["model_kwargs"] != source.config["model_kwargs"]:
                raise ValueError(f"{key}: model_kwargs must match {exp.dependency} exactly")
            if exp.config.get("phase1_experiment") != source.config.get("phase1_experiment"):
                raise ValueError(f"{key}: phase1_experiment must match {exp.dependency}")
            if exp.runner == "recon" and exp.config.get("recon_kwargs", {}).get("with_aux"):
                if not source.config.get("aux_kwargs", {}).get("with_aux", True):
                    raise ValueError(f"{key}: auxiliary reconstruction needs a phase-one auxiliary head")
                exp.config["aux_kwargs"] = copy.deepcopy(source.config["aux_kwargs"])
            exp.config["pretrained_from"] = str(checkpoint_path(source, root_dir, version))
        visiting.remove(key)
        visited.add(key)
        plan.append(exp)

    for key in selected:
        visit(key)
    return plan, root_dir


def _function(reference):
    module, name = reference.split(":")
    return getattr(importlib.import_module(module), name)


def train_experiment(exp, env_path, version):
    args = SimpleNamespace(config_env=str(env_path), config_exp=str(exp.config_path),
                           fname="psm", version=version)
    trainer = _function(RUNNERS[exp.runner][0])
    if exp.runner in ("ae", "vae"):
        return trainer(exp.runner, args, update_dictionary=exp.config)
    return trainer(args, update_dictionary=exp.config)


def _build_model(exp, config):
    return _function(RUNNERS[exp.runner][1])(config)


def score_experiment(exp, env_path, version, *, fname="psm"):
    """All frameworks use validation-selected weights and the same evaluation."""
    import numpy as np
    import torch

    from data.jepa_dataset import JEPADataset
    from metrics.metrics import combine_all_evaluation_scores
    from utils.common_config import get_jepa_datasets
    from utils.reconstruction_baselines import (evaluate_from_scores, score_both)
    from utils.scoring import covered_evaluation_view, evaluation_options
    from utils.trainer import Trainer

    p = create_config(env_path, exp.config_path, fname, version,
                      update_dictionary=exp.config)
    requested_device = str(p.get("device", "cpu"))
    device = torch.device(requested_device if not requested_device.startswith("cuda")
                          or torch.cuda.is_available() else "cpu")
    model = _build_model(exp, p).to(device)
    weights = Path(p.get("score_checkpoint") or Path(p["jepa_dir"]) / RUNNERS[exp.runner][2])
    if not weights.is_file():
        raise FileNotFoundError(f"validation-selected checkpoint missing: {weights}")
    Trainer.load_weights(str(weights), model, strict=True)
    model.eval()
    if exp.runner == "cross_attention":
        from lewm_cross_attention import scoring_options

        options = scoring_options(p, model)
    else:
        options = evaluation_options(p)
    window, stride = options.pop("wsz"), options.pop("stride")
    _, val_dataset = get_jepa_datasets(p)
    test_dataset = JEPADataset(p, train=False)

    class InputResolutionScores:
        """LEWM's native map is latent-resolution when its encoder downsamples."""

        def eval(self):
            model.eval()
            return self

        def score(self, x):
            output = model.score(x)
            if exp.runner == "lewm":
                step = int(model.level_strides[0])
                output["fused"] = output["fused"].repeat_interleave(step, dim=-1)
            return output

    scorer_model = InputResolutionScores()
    batch_size = int(p.get("score_batch_size", p["batch_size"]))
    clean = score_both(scorer_model, val_dataset.series, window, stride,
                       batch_size, device, **options)
    test = score_both(scorer_model, test_dataset.series, window, stride,
                      batch_size, device, **options)
    quantile = float(p.get("calibration_kwargs", {}).get(
        "quantile", p.get("calibration_quantile", 0.995)))
    evaluation = evaluate_from_scores(clean, test, test_dataset.targets,
                                      {**p, "calibration_quantile": quantile})
    thresholds = {name: values["calibrated"]["threshold"]
                  for name, values in evaluation.items()}
    scores, labels, _, _ = covered_evaluation_view(
        test["timeseries_scores"], test_dataset.targets, test["starts"],
        test["ends"], test["cover_counts"])
    metric_dict = combine_all_evaluation_scores(
        (scores > thresholds["timeseries"]).astype(np.int64), labels,
        int(p.get("eval_window_size", 100)))
    report = {
        "framework": exp.framework, "experiment_name": exp.name,
        "experiment_key": exp.key,
        "phase1_experiment": exp.config.get("phase1_experiment"),
        "run": version, "dataset": p["train_db_name"], "runner": exp.runner,
        "selection": {"weights": str(weights), "source": "validation loss"},
        "calibration_source": "held-out clean train tail",
        "evaluation": evaluation,
        "honest": {k: float(v) for k, v in metric_dict.items() if not k.startswith("pa_")},
        "point_adjust_comparability": {k[3:]: float(v) for k, v in metric_dict.items()
                                      if k.startswith("pa_")},
        "coverage": {"scored_timesteps": int((test["cover_counts"] > 0).sum()),
                     "total_timesteps": len(test_dataset.series),
                     "first_scored_timestep": int(test["starts"].min()),
                     "input_window": window,
                     "output_window": int(test["ends"][0] - test["starts"][0])},
    }
    calibration = {"source": "held-out clean train tail", "quantile": quantile,
                   "window_threshold": thresholds["window"],
                   "timeseries_threshold": thresholds["timeseries"]}
    for path, payload in ((p["metrics_path"], report), (p["calibration_path"], calibration)):
        with open(path, "w") as stream:
            json.dump(payload, stream, indent=2)
    np.savez_compressed(
        p["scores_path"], window_scores=test["window_scores"],
        timeseries_scores=test["timeseries_scores"],
        start_idxs=test["starts"], end_idxs=test["ends"],
        input_start_idxs=test["input_starts"], input_end_idxs=test["input_ends"],
        cover_counts=test["cover_counts"],
        window_predictions=test["window_scores"] > thresholds["window"],
        timeseries_predictions=(test["timeseries_scores"] > thresholds["timeseries"])
                              & (test["cover_counts"] > 0),
        window_labels=np.asarray([np.any(test_dataset.targets[s:e])
                                  for s, e in zip(test["starts"], test["ends"])], dtype=np.int64),
        timestep_labels=test_dataset.targets)
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(str(Path(p["jepa_dir"]) / "suite_evaluation")) as writer:
        for procedure, values in evaluation.items():
            for source in ("calibrated", "oracle"):
                for name, value in values[source].items():
                    writer.add_scalar(f"test/{procedure}/{source}/{name}", value, 0)
            for name in ("vus_pr", "vus_roc"):
                writer.add_scalar(f"test/{procedure}/{name}", values[name], 0)
    return report


def _save_resolved_config(exp, directory):
    path = directory / "resolved_config.yml"
    if path.exists():
        old = _load_yaml(path)
        new = copy.deepcopy(exp.config)
        # Epoch extensions and runtime placement can change on resume.
        for config in (old, new):
            for key in ("epochs", "device", "amp", "num_workers", "score_batch_size"):
                config.pop(key, None)
        if old != new:
            raise ValueError(f"{exp.key}: configuration changed in an existing run; use a new experiment_name or --version")
    path.write_text(yaml.safe_dump(exp.config, sort_keys=False))


SUMMARY_FIELDS = ["framework", "experiment_name", "experiment_key", "phase1_experiment", "run", "status", "runner",
                  "timeseries_f1", "window_f1", "vus_pr", "vus_roc",
                  "scored_timesteps", "total_timesteps", "output_window",
                  "seconds", "checkpoint", "metrics_path", "error"]


def write_summary(directory, rows):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(rows, indent=2))
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_suite(plan, root_dir, env_path, version, *, score_only=False,
              fail_fast=False):
    """Run dependencies once, save partial summaries, and isolate failed experiments."""
    rows, statuses = [], {}
    summary_dir = Path(root_dir) / "psm" / "summaries" / version
    summary_path = summary_dir / "summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else []
    combined = {row.get("experiment_key", f"{row['framework']}/{row['experiment_name']}"): row for row in existing}
    for exp in plan:
        directory = experiment_dir(exp, root_dir, version)
        row = {"framework": exp.framework, "experiment_name": exp.name,
               "experiment_key": exp.key, "phase1_experiment": exp.config.get("phase1_experiment"),
               "runner": exp.runner, "run": version,
               "checkpoint": str(checkpoint_path(exp, root_dir, version)),
               "metrics_path": str(directory / "metrics.json")}
        started = time.monotonic()
        print(f"{'SCORE' if score_only else 'RUN'} {exp.key} -> {directory}", flush=True)
        try:
            if exp.dependency and statuses.get(exp.dependency) != "completed":
                row.update(status="blocked", error=f"source experiment {exp.dependency} failed")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                _save_resolved_config(exp, directory)
                if not score_only:
                    train_experiment(exp, env_path, version)
                report = score_experiment(exp, env_path, version)
                values = report["evaluation"]
                row.update(status="completed",
                           timeseries_f1=values["timeseries"]["calibrated"]["f1_no_pa"],
                           window_f1=values["window"]["calibrated"]["f1_no_pa"],
                           vus_pr=values["timeseries"]["vus_pr"],
                           vus_roc=values["timeseries"]["vus_roc"],
                           **{k: report["coverage"][k] for k in
                              ("scored_timesteps", "total_timesteps", "output_window")})
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            if directory.exists():
                (directory / "suite_error.txt").write_text(traceback.format_exc())
            print(f"FAILED {exp.key}: {exc}", flush=True)
        row["seconds"] = round(time.monotonic() - started, 2)
        statuses[exp.key] = row["status"]
        rows.append(row)
        combined[exp.key] = row
        write_summary(summary_dir, list(combined.values()))
        if fail_fast and row["status"] != "completed":
            break
    print("\nExperiment                                      Status       Series F1  Window F1  VUS PR")
    for row in combined.values():
        values = "  ".join(f"{row[k]:.4f}" if k in row else "   -  "
                            for k in ("timeseries_f1", "window_f1", "vus_pr"))
        print(f"{row.get('experiment_key', row['framework']+'/'+row['experiment_name']):<47} {row['status']:<12} {values}")
    print(f"Summary: {summary_dir / 'summary.csv'}", flush=True)
    return rows
