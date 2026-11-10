"""Resolve dataset selections once and run isolated, resumable native jobs."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import yaml

from utils.config import (_training_sections, create_config, experiment_base_dir,
                          load_experiment_config, DEFAULT_VERSION)


def _collection(value):
    aliases = {"tsb_ad_u": "tsb-ad-u", "tsb_ad_m": "tsb-ad-m",
               "tsb-ad-u": "tsb-ad-u", "tsb-ad-m": "tsb-ad-m"}
    return aliases.get(str(value).lower(), str(value).lower())


def _relative_file(root, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("every selected series requires a nonempty filename")
    relative = Path(value.strip())
    resolved = (root / relative).resolve()
    if relative.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError("series filenames must stay inside the collection root")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return relative.as_posix(), resolved


def _manifest(path, root):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if "file_name" not in (reader.fieldnames or []):
            raise ValueError("TSB manifest requires file_name")
        entries, seen = [], set()
        for row in reader:
            name, resolved = _relative_file(root, row.get("file_name"))
            if resolved in seen:
                raise ValueError(f"duplicate manifest series: {name}")
            seen.add(resolved)
            prefix = row.get("train_end", "")
            if prefix is not None and str(prefix).strip():
                if not str(prefix).strip().isdigit():
                    raise ValueError(f"{name}: train_end must be an integer")
                prefix = int(prefix)
            else:
                prefix = None
            entries.append((name, prefix))
    if not entries:
        raise ValueError("manifest contains no series")
    return entries


def _output_series(name):
    path = Path(name)
    return name if len(path.parts) == 1 else path.name + "-" + hashlib.sha256(name.encode()).hexdigest()[:10]


def _source_paths(config, root_dir):
    roles = (config.get("lewm_modules", {}) if config["framework"] == "lewm"
             else config.get("feature_modules", {}))
    for role in ("encoder", "backbone"):
        options = roles.get(role)
        if not isinstance(options, dict):
            continue
        source = options.pop("source", None)
        value = options.get("pretrained_from", source)
        if value is None:
            continue
        if isinstance(value, dict):
            source = dict(value)
            framework = source.pop("framework", "lewm")
            experiment = source.pop("experiment_name", source.pop("experiment", "masked_frequency"))
            version = source.pop("version", config.get("version", DEFAULT_VERSION))
            fname = source.pop("fname", config["fname"])
            root = source.pop("root_dir", root_dir)
            checkpoint = source.pop("checkpoint", None)
            if source:
                raise ValueError(f"unknown source options: {sorted(source)}")
            source_config = {**config, "framework": framework, "experiment_name": experiment,
                             "output_series": _output_series(fname)}
            directory = Path(experiment_base_dir(root, source_config, fname, version)) / "jepa"
            candidates = ([directory / checkpoint] if checkpoint else
                          [directory / "best_validation_loss.pth.tar",
                           directory / "best_train_loss.pth.tar", directory / "model.pth.tar"])
            value = str(next((path for path in candidates if path.is_file()), candidates[-1]))
        elif not isinstance(value, str):
            raise ValueError("pretrained_from must be a checkpoint path or source mapping")
        value = value.format(dataset=config["train_db_name"], collection=config.get("dataset_collection", config["train_db_name"]),
                             fname=config["fname"], version=config.get("version", DEFAULT_VERSION))
        options["pretrained_from"] = value


def preflight_job(config):
    """Validate actual readers, source weights, model geometry and objective."""
    import torch
    from data.jepa_dataset import JEPADataset, JEPACorpusDataset
    from models import get_model
    from train import check_model_loss, run_contract
    from utils.common_config import get_jepa_datasets, get_criterion
    from utils.model_contract import prepare_feature_source_config, initialize_feature_source
    from utils.randomness import isolated_rng
    from utils.trainer import Trainer

    config = deepcopy(config)
    train_data, val_data = get_jepa_datasets(config)
    if not len(train_data):
        raise ValueError("training selection contains no windows")
    if isinstance(train_data, JEPACorpusDataset):
        for name in train_data.machine_files:
            JEPADataset({**config, "fname": name}, train=False)
    else:
        JEPADataset(config, train=False)
    config["validation_available"] = bool(len(val_data))
    config["_feature_schema"] = list(train_data.feature_schema)
    config["_preprocessing"] = train_data.preprocessing
    config["_run_contract"] = run_contract(config)
    own_path = Path(config.get("score_checkpoint") or config["jepa_model"]) if config.get("stage") == "score" else Path(config["jepa_checkpoint"])
    if config.get("stage") == "score" and not own_path.is_file():
        raise FileNotFoundError(f"selected scoring checkpoint does not exist: {own_path}")
    saved = torch.load(own_path, map_location="cpu", weights_only=False) if own_path.is_file() else None
    with isolated_rng(int(config.get("seed", 4))):
        config = prepare_feature_source_config(config, train_data.feature_schema, resume_checkpoint=saved)
        model = get_model(config, input_channels=len(train_data.feature_schema))
        if saved is not None:
            Trainer.validate_run_contract(config, saved)
            model.load_state_dict(saved["model"], strict=True)
        elif not config.get("initialize_from"):
            initialize_feature_source(model, config, train_data.feature_schema)
        else:
            Trainer.initialize_weights(config["initialize_from"], model)
        batch = torch.from_numpy(__import__("numpy").stack([
            train_data[index]["ts"] for index in range(min(2, len(train_data)))]))
        check_model_loss(model, get_criterion(config), {"ts": batch}, torch.device("cpu"))
    return config


def plan_jobs(config_exp, config_env="configs/env.yml", *, dataset=None, fname=None,
              file_list=None, dataset_root=None, version=None, output_root=None, overrides=None):
    """Return fully preflighted configs; no training starts until every job passes."""
    if fname is not None and file_list is not None:
        raise ValueError("fname and file_list are mutually exclusive")
    base = _training_sections(load_experiment_config(config_exp))
    collection = _collection(dataset or base.get("dataset_collection", base["train_db_name"]))
    if collection not in ("psm", "smd", "tsb-ad-u", "tsb-ad-m", "tsb_ad", "synthetic"):
        raise ValueError(f"unsupported dataset collection {collection}")
    internal = "tsb_ad" if collection.startswith("tsb-ad-") else collection
    base.update(train_db_name=internal, val_db_name=internal, dataset_collection=collection)
    if dataset_root is not None:
        base["dataset_root"] = str(dataset_root)
    elif dataset is not None and internal != _training_sections(load_experiment_config(config_exp))["train_db_name"]:
        base.pop("dataset_root", None)
    if internal != "synthetic":
        defaults = {"psm": "datasets/PSM", "smd": "datasets/SMD",
                    "tsb-ad-u": "datasets/TSB-AD-U", "tsb-ad-m": "datasets/TSB-AD-M"}
        root = Path(base.get("dataset_root") or defaults.get(collection, "datasets/TSB-AD")).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {root}")
        base["dataset_root"] = str(root)
    else:
        root = Path(".")
    selection = fname or base.get("fname")
    if internal == "psm":
        if fname is not None or file_list is not None:
            raise ValueError("PSM takes no filename or file list")
        entries = [("psm", None)]
        selection = "psm"
    elif internal == "smd":
        if file_list is not None:
            raise ValueError("file-list is only supported for TSB collections")
        machines = sorted(path.name for path in (root / "train").glob("machine-*.txt"))
        if selection in ("all", "all-one"):
            if not machines:
                raise ValueError("SMD selection contains no machine files")
            entries = [(name, None) for name in machines] if selection == "all" else [("all-one", None)]
            if selection == "all-one":
                base["machine_files"] = machines
        else:
            if not selection:
                raise ValueError("SMD requires a machine filename, all, or all-one")
            entries = [(selection, None)]
    elif internal == "tsb_ad":
        entries = _manifest(file_list, root) if file_list else [(_relative_file(root, selection)[0], base.get("tsb_train_end"))]
        selection = str(Path(file_list).resolve()) if file_list else selection
        base["tsb_evaluation_scope"] = "test"
    else:
        if file_list:
            raise ValueError("synthetic does not accept file lists")
        entries = [("synthetic", None)]
        selection = "synthetic"
    with Path(config_env).open() as stream:
        root_dir = str(output_root or yaml.safe_load(stream)["root_dir"])
    version = version or DEFAULT_VERSION
    suite_identity = {"collection": collection, "selection": selection,
                      "config_exp": str(Path(config_exp).resolve()), "version": version,
                      "stage": (overrides or {}).get("stage", base.get("stage", "train")),
                      "score_checkpoint": (overrides or {}).get("score_checkpoint")}
    digest = hashlib.sha256(json.dumps(suite_identity, sort_keys=True).encode()).hexdigest()[:12]
    suite_path = Path(root_dir) / collection / "suites" / base["framework"] / base.get("experiment_name", "default") / version / f"{digest}.json"
    configs = []
    for name, prefix in entries:
        updates = deepcopy(base)
        updates.update(overrides or {})
        updates.update(fname=name, output_series=_output_series(name))
        if prefix is not None:
            updates["tsb_train_end"] = prefix
        directory = Path(experiment_base_dir(root_dir, updates, name, version)) / "jepa"
        config = create_config(config_env, config_exp, name, version,
                               update_dictionary=updates, output_dir=directory)
        config.update(root_dir=root_dir, suite_manifest=str(suite_path),
                      config_exp=str(Path(config_exp).resolve()), config_env=str(Path(config_env).resolve()))
        _source_paths(config, root_dir)
        configs.append(preflight_job(config))
    return configs


def _worker(config):
    import os
    import torch
    from train import run
    torch.set_num_threads(int(config.get("cpu_threads", 1)))
    started = time.time()
    try:
        result = run(config)
        return {"status": "completed", "result": result, "pid": os.getpid(),
                "elapsed_seconds": time.time() - started}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), "pid": os.getpid(),
                "elapsed_seconds": time.time() - started}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(path)


def run_jobs(configs, *, jobs=None, manifest_path=None):
    """The parent records every result and skips already completed native runs."""
    if not configs:
        raise ValueError("job plan is empty")
    jobs = int(jobs if jobs is not None else configs[0].get("jobs", 1))
    if jobs < 1:
        raise ValueError("jobs must be positive")
    path = Path(manifest_path or configs[0]["suite_manifest"])
    previous = json.loads(path.read_text()) if path.is_file() else {}
    old = {item["job_id"]: item for item in previous.get("jobs", [])}
    rows, pending = [], []
    for config in configs:
        job_id = str(Path(config["jepa_dir"]).resolve())
        row = {"job_id": job_id, "fname": config["fname"], "seed": int(config.get("seed", 4)),
               "collection": config.get("dataset_collection", config["train_db_name"]),
               "resolved_config": config, "checkpoint": config["jepa_checkpoint"],
               "metrics": config["metrics_path"], "status": "pending"}
        saved = old.get(job_id)
        if saved and saved.get("status") == "completed":
            from train import run_contract
            if run_contract(saved["resolved_config"]) != run_contract(config):
                raise ValueError("completed suite job configuration changed; use a new version")
            from utils.evaluation_schedule import evaluation_interval
            required = ([Path(config[key]) for key in ("metrics_path", "calibration_path")] if config.get("stage") == "score" else
                        [Path(config["jepa_checkpoint"]), Path(config["jepa_model"])])
            if config.get("stage") != "score" and evaluation_interval(config):
                required.extend(Path(config[key]) for key in ("metrics_path", "calibration_path"))
            if config.get("stage") == "score" or evaluation_interval(config):
                if config.get("fname") == "all-one":
                    required.extend(Path(config["jepa_dir"]) / "machines" / name / artifact
                                    for name in config.get("machine_files", [])
                                    for artifact in ("metrics.json", "calibration.json", "scores.npz"))
                else:
                    required.append(Path(config["scores_path"]))
            if (all(artifact.is_file() for artifact in required)
                    and int(saved["resolved_config"].get("epochs", 1)) >= int(config.get("epochs", 1))):
                row.update({key: saved[key] for key in ("status", "pid", "elapsed_seconds", "result") if key in saved})
                row["skipped_completed"] = True
            else:
                pending.append((config, row))
        else:
            pending.append((config, row))
        rows.append(row)
    manifest = {"version": 1, "concurrency": jobs, "jobs": rows}
    _atomic_json(path, manifest)
    if jobs == 1:
        for config, row in pending:
            row["status"] = "running"
            _atomic_json(path, manifest)
            row.update(_worker(config))
            _atomic_json(path, manifest)
    elif pending:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as executor:
            futures = {executor.submit(_worker, config): row for config, row in pending}
            for row in futures.values():
                row["status"] = "running"
            _atomic_json(path, manifest)
            for future in as_completed(futures):
                row = futures[future]
                try:
                    row.update(future.result())
                except Exception as exc:
                    row.update(status="failed", error=f"worker process failed: {exc}")
                _atomic_json(path, manifest)
    return manifest
