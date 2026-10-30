"""Run native experiments against TSB-AD while retaining training artifacts."""

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import yaml

from data.tsb_ad import read_tsb_series
from utils.config import DEFAULT_VERSION, merge_config, validate_experiment_name
from utils.experiment_suite import (RUNNERS, _build_model, build_plan, checkpoint_path,
                                    experiment_dir, train_experiment)


def select_files(dataset_root, *, files=None, file_list=None):
    """Accept the official file_name manifest or explicit series paths."""
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise ValueError(f"dataset directory does not exist: {root}")
    if files and file_list:
        raise ValueError("select --files or --file-list")
    if file_list:
        with Path(file_list).open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "file_name" not in reader.fieldnames:
                raise ValueError("TSB-AD file lists require a file_name column")
            names = [row["file_name"] for row in reader]
    else:
        names = files if files else sorted(str(p.relative_to(root)) for p in root.rglob("*.csv"))
    paths = [(root / name).resolve() for name in names]
    if not paths:
        raise ValueError(f"no benchmark CSV files selected in {root}")
    if len(set(p.name for p in paths)) != len(paths):
        raise ValueError("series filenames must be unique within a benchmark run")
    for path in paths:
        if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() != ".csv":
            raise ValueError(f"expected a CSV inside {root}: {path}")
        validate_experiment_name(path.name, "series filename")
    return paths


def adapt_geometry(config, train_end, policy="shrink", *, runner=None):
    """Resolve short benchmark prefixes without crossing train/validation splits."""
    config = merge_config(config, {})
    fraction = float(config.get("val_fraction", 0.2))
    if not 0 < fraction < 1:
        raise ValueError("val_fraction must lie strictly between 0 and 1")
    cut = int(train_end * (1 - fraction))
    available = min(cut, train_end - cut)
    strides = config.get("model_kwargs", {}).get("enc_strides", [1])
    step = int(np.prod(strides))
    if step < 1:
        raise ValueError("encoder strides must be positive")
    lattice = step
    crops = config.get("cross_attention_kwargs", {})
    if runner == "cross_attention" and crops.get("context_crop") is None and crops.get("target_crop") is None:
        lattice = 2 * step  # half splits require an even number of feature tokens
    requested = int(config["wsz"])
    window = requested
    if policy == "shrink":
        window = min(window, available) // lattice * lattice
    elif policy != "strict":
        raise ValueError("window policy must be shrink or strict")
    minimum = step * (8 if runner == "lewm" and config.get("predictor_kwargs", {}).get("domain") == "frequency" else 2)
    if window < minimum or window > available or window % lattice:
        raise ValueError(f"prefix {train_end} provides train/validation lengths {cut}/{train_end-cut}; "
                         f"window {window} must fit both, be divisible by {lattice}, and be >= {minimum}. "
                         "Adjust --window, --val-fraction, or the model configuration.")
    stride = min(int(config["stride"]), window)
    if stride < 1:
        raise ValueError("stride must be positive")
    windows = (cut - window) // stride + 1
    config.update(wsz=window, stride=stride, batch_size=min(int(config["batch_size"]), windows))
    if config["batch_size"] < 1:
        raise ValueError("batch_size must be positive")
    # Explicit evaluation geometry remains authoritative. Otherwise use the
    # resolved training geometry, including an off-grid final window.
    config.setdefault("evaluation", {})
    config["evaluation"].setdefault("include_last_window", True)
    return config


def build_benchmark_plan(manifest, env_path, version, series, root, *,
                         overrides=None, frameworks=None, experiments=None,
                         epochs=None, device=None, window_policy="shrink"):
    settings = merge_config(overrides or {}, {
        "model_kwargs": {"in_channels": series.channels},
        "benchmark_protocol": "tsb_ad", "eval_every": 0,
        "decision_trace": False, "stage_a": {"corpus": "single"},
    })
    plan, _ = build_plan(manifest, env_path, version, frameworks=frameworks,
                         experiments=experiments, epochs=epochs, device=device,
                         dataset="tsb_ad", fname=series.path.name,
                         dataset_root=str(series.path.parent), config_overrides=settings,
                         root_dir_override=str(root))
    resolved = []
    for exp in plan:
        try:
            config = adapt_geometry(exp.config, series.train_end, window_policy, runner=exp.runner)
        except ValueError as exc:
            # One unsuitable model must not prevent compatible arms from
            # running on the same short-prefix series.
            config = {**exp.config, "benchmark_planning_error": str(exc)}
        resolved.append(replace(exp, config=config))
    return resolved


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def record_run(exp, root, version, series, *, score_only=False):
    """Reject accidentally resuming a checkpoint with a different experiment."""
    directory = experiment_dir(exp, root, version)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "benchmark_run.json"
    mutable = {"epochs", "device", "num_workers", "score_batch_size", "resume_learning_rate",
               "initialize_from", "optimizer_kwargs", "score_checkpoint", "decision_trace"}
    signature = {k: v for k, v in exp.config.items() if k not in mutable}
    # The explicit resume LR is the only optimizer setting allowed to change.
    signature["optimizer_kwargs"] = {k: v for k, v in exp.config.get("optimizer_kwargs", {}).items()
                                      if k != "lr"}
    identity = {"config": signature, "data_sha256": hashlib.sha256(series.path.read_bytes()).hexdigest(),
                "train_end": series.train_end}
    initialization = exp.config.get("initialize_from")
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity"] != identity:
            raise ValueError(f"run configuration or input changed at {directory}; use a new --version/--fresh")
        initialization = previous.get("initialization_source", initialization)
    elif score_only:
        raise FileNotFoundError(f"benchmark run metadata missing: {path}")
    _json(path, {"identity": identity, "experiment": exp.key, "file": series.path.name,
                 "source_config": str(exp.config_path), "version": version,
                 "resolved_config": exp.config, "initialization_source": initialization})
    (directory / "resolved_config.yml").write_text(yaml.safe_dump(exp.config, sort_keys=False))
    return directory


class InputResolutionModel:
    """Map LeWM's latent token scores back to the input timestep grid."""

    def __init__(self, model, runner):
        self.model, self.runner = model, runner

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def score(self, x):
        output = self.model.score(x)
        if self.runner == "lewm":
            step = int(self.model.level_strides[0])
            output = {**output, "fused": output["fused"].repeat_interleave(step, dim=-1)[..., :x.shape[-1]]}
            for key in ("levels", "signals"):
                output[key] = {name: values.repeat_interleave(step, dim=-1)[..., :x.shape[-1]]
                               for name, values in output[key].items()}
        return output


def aligned_scores(result):
    """Preserve TSB's full timeline; interpolate gaps and extend boundary scores."""
    scores = np.asarray(result["scores"], dtype=np.float64)
    covered = np.flatnonzero(result["cover_counts"] > 0)
    if not len(covered) or not np.isfinite(scores[covered]).all():
        raise ValueError("model produced empty or nonfinite covered scores")
    return np.interp(np.arange(len(scores)), covered, scores[covered])


def score_benchmark(exp, env_path, root, version, series, *, metrics="deferred",
                    tsb_ad_root=None, sliding_window=None, trace=True):
    started = time.perf_counter()
    import torch
    from data.jepa_dataset import JEPADataset
    from sklearn.metrics import f1_score, precision_score, recall_score
    from utils.config import create_config
    from utils.decision_trace import write_final_decision_trace
    from utils.scoring import Scorer, evaluation_options
    from utils.trainer import Trainer
    from utils.utils import Logger

    directory = experiment_dir(exp, root, version)
    p = create_config(env_path, exp.config_path, exp.config["fname"], version,
                      update_dictionary=exp.config, output_dir=str(directory))
    requested = str(p.get("device", "cpu"))
    device = torch.device(requested if not requested.startswith("cuda") or torch.cuda.is_available() else "cpu")
    model = _build_model(exp, p).to(device)
    weights = Path(p.get("score_checkpoint") or checkpoint_path(exp, root, version))
    if not weights.is_file():
        raise FileNotFoundError(f"validation-selected checkpoint missing: {weights}")
    Trainer.load_weights(str(weights), model, strict=True)
    model.eval()
    if exp.runner == "cross_attention":
        from lewm_cross_attention import scoring_options
        geometry = p.get("evaluation", {})
        if "context_crop" in geometry or "target_crop" in geometry:
            if not {"context_crop", "target_crop"} <= geometry.keys():
                raise ValueError("evaluation requires both context_crop and target_crop")
            model.context_crop = model._interval(geometry["context_crop"])
            model.target_crop = model._interval(geometry["target_crop"])
        options = scoring_options(p, model)
    else:
        options = evaluation_options(p)
        if {"context_crop", "target_crop"} & (p.get("evaluation", {}) or {}).keys():
            raise ValueError("evaluation crops require cross_attention")
    window, stride = options.pop("wsz"), options.pop("stride")
    train_data = JEPADataset(p, train=True)
    clean = JEPADataset.validation_split(train_data, p)
    test = JEPADataset(p, train=False)
    wrapped = InputResolutionModel(model, exp.runner)
    scorer = Scorer(wrapped, device, int(p.get("score_batch_size", p["batch_size"])))
    clean_result = scorer.score_series(clean.series, window, stride, **options)
    test_result = scorer.score_series(test.series, window, stride, **options)
    scores = aligned_scores(test_result)
    score_seconds = time.perf_counter() - started
    if scores.shape != series.labels.shape:
        raise ValueError("benchmark scores must match the full input timeline")
    quantile = float(p.get("calibration_kwargs", {}).get("quantile", p.get("calibration_quantile", 0.995)))
    if not 0 < quantile < 1:
        raise ValueError("calibration quantile must lie strictly between 0 and 1")
    threshold = float(np.quantile(clean_result["scores"][clean_result["cover_counts"] > 0], quantile))
    predictions = (scores > threshold).astype(np.int64)
    # Raw scores are the upstream benchmark artifact. Labels and calibration
    # predictions are separate, available to a compatible evaluator process.
    score_path = directory / "benchmark_scores.npy"
    np.save(score_path, scores)
    inputs_path = directory / "evaluation_inputs.npz"
    np.savez_compressed(inputs_path, labels=series.labels, first_channel=series.data[:, 0],
                        predictions=predictions)
    np.savez_compressed(directory / "scores.npz", scores=scores, labels=series.labels,
                        predictions=predictions, cover_counts=test_result["cover_counts"],
                        starts=test_result["start_idxs"], ends=test_result["end_idxs"],
                        **{k.replace("/", "_"): v for k, v in test_result["channels"].items()})
    calibration = {"threshold": threshold, "quantile": quantile,
                   "source": "declared_train_prefix_validation_tail", "operator": "gt",
                   "normalization_mean": train_data.mean.tolist(), "normalization_std": train_data.std.tolist()}
    _json(directory / "calibration.json", calibration)
    report = {"protocol": "TSB-AD semisupervised: fit declared prefix, score full series",
              "train_end": series.train_end, "checkpoint": str(weights.resolve()),
              "score_length": len(scores), "coverage": {
                  "covered": int(np.count_nonzero(test_result["cover_counts"])),
                  "uncovered": int(np.count_nonzero(test_result["cover_counts"] == 0)),
                  "fill_policy": "linear interpolation of internal gaps; nearest boundary extension"},
              "calibrated": {"f1": float(f1_score(series.labels, predictions, zero_division=0)),
                             "precision": float(precision_score(series.labels, predictions, zero_division=0)),
                             "recall": float(recall_score(series.labels, predictions, zero_division=0)),
                             **calibration},
              "prefix_anomalies": int(np.count_nonzero(series.labels[:series.train_end])),
              "metric_status": "deferred"}
    logger = Logger(version, verbose=3, file_path=str(directory), file_name="benchmark", delete_files=False)
    try:
        logger.log(f"TSB-AD {series.path.name}: {exp.key}; checkpoint={weights}")
        if metrics == "tsb-ad":
            from utils.tsb_ad_metrics import evaluate_scores
            report["tsb_ad"] = evaluate_scores(scores, series.labels, series.data[:, 0],
                                               tsb_ad_root=tsb_ad_root, sliding_window=sliding_window)
            report["metric_status"] = "completed"
            _json(directory / "tsb_metrics.json", report["tsb_ad"])
        for section in ("calibrated", "tsb_ad"):
            for name, value in report.get(section, {}).items():
                if isinstance(value, (int, float)):
                    logger.scalar_summary(f"TSB-AD/{section}", name, value, int(p["epochs"]))
        if trace:
            report["decision_trace"] = write_final_decision_trace(
                wrapped, device, test.series, {**p, "decision_trace": {"max_windows": 256}}, threshold,
                options={"wsz": window, "stride": stride, **options},
                threshold_operator="gt", labels=series.labels)
        _json(directory / "metrics.json", report)
        logger.log(f"Calibrated metrics: {report['calibrated']}")
        logger.log(f"Official metrics: {report.get('tsb_ad', 'deferred; run evaluate_tsb_ad.py')}")
    finally:
        logger.finalize()
    return {"score_path": str(score_path.resolve()), "evaluation_inputs_path": str(inputs_path.resolve()),
            "metrics_path": str((directory / "metrics.json").resolve()), "metric_status": report["metric_status"],
            "score_seconds": score_seconds, "tsb_ad": report.get("tsb_ad")}


def write_metric_tables(run_dir, rows):
    """Write the official file,Time,metric CSV layout for inline evaluations."""
    from utils.tsb_ad_metrics import METRIC_NAMES
    algorithms = {row["algorithm"] for row in rows if row.get("tsb_ad")}
    for algorithm in algorithms:
        directory = run_dir / "benchmark_metrics"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{algorithm}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["file", "Time", *METRIC_NAMES])
            writer.writeheader()
            for row in rows:
                if row.get("algorithm") == algorithm and row.get("tsb_ad"):
                    writer.writerow({"file": row["file"], "Time": row["detector_seconds"],
                                     **{key: row["tsb_ad"].get(key) for key in METRIC_NAMES}})


def run_benchmark(plans, env_path, root, version, *, score_only=False, fail_fast=False,
                  metrics="deferred", tsb_ad_root=None, sliding_window=None, trace=True):
    """Persist failures and continue independently, retaining every native run."""
    root = Path(root).resolve()
    run_dir = root / "benchmark_runs" / version
    run_dir.mkdir(parents=True, exist_ok=True)
    env = run_dir / "env.yml"
    env.write_text(yaml.safe_dump({"root_dir": str(root)}))
    rows = []
    summary_path = run_dir / "summary.json"
    saved = {}
    if summary_path.is_file():
        saved = {(row["file"], row["experiment"]): row
                 for row in json.loads(summary_path.read_text())}

    def persist(row):
        saved[(row["file"], row["experiment"])] = row
        _json(summary_path, list(saved.values()))
        if metrics == "tsb-ad":
            write_metric_tables(run_dir, list(saved.values()))

    for series, plan in plans:
        if series is None:
            rows.append(plan)  # a series-level loading/planning failure
            persist(plan)
            if fail_fast:
                return rows
            continue
        failed_dependencies = set()
        for exp in plan:
            started = time.perf_counter()
            directory = experiment_dir(exp, root, version)
            row = {"experiment": exp.key, "algorithm": exp.key.replace("/", "__"),
                   "file": series.path.name, "directory": str(directory), "status": "running"}
            try:
                if exp.config.get("benchmark_planning_error"):
                    raise ValueError(exp.config["benchmark_planning_error"])
                if exp.dependency in failed_dependencies:
                    raise RuntimeError(f"dependency failed: {exp.dependency}")
                record_run(exp, root, version, series, score_only=score_only)
                _json(directory / "benchmark_result.json", row)
                training_started = time.perf_counter()
                if not score_only:
                    train_experiment(exp, str(env), version)
                row["fit_seconds"] = time.perf_counter() - training_started
                row.update(score_benchmark(exp, str(env), root, version, series, metrics=metrics,
                                           tsb_ad_root=tsb_ad_root, sliding_window=sliding_window, trace=trace))
                row["detector_seconds"] = row["fit_seconds"] + row["score_seconds"]
                exported = run_dir / "benchmark_scores" / row["algorithm"] / f"{series.path.stem}.npy"
                exported.parent.mkdir(parents=True, exist_ok=True)
                np.save(exported, np.load(row["score_path"], allow_pickle=False))
                row["upstream_score_path"] = str(exported)
                row["status"] = "completed"
            except Exception as exc:
                failed_dependencies.add(exp.key)
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "benchmark_error.txt").write_text(traceback.format_exc())
                print(f"FAILED {series.path.name} {exp.key}: {exc}", flush=True)
                from utils.utils import Logger
                logger = Logger(version, verbose=2, file_path=str(directory),
                                file_name="benchmark", use_tensorboard=False, delete_files=False)
                logger.error(row["error"])
                logger.finalize()
            row["elapsed_seconds"] = time.perf_counter() - started
            _json(directory / "benchmark_result.json", row)
            rows.append(row)
            persist(row)
            if row["status"] == "failed" and fail_fast:
                return rows
    return rows
