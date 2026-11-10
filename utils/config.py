import os
from copy import deepcopy
from pathlib import Path
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing as mkdir


DEFAULT_VERSION = "default"


def validate_experiment_name(value, field):
    """Names are one directory component, never a path."""
    import re

    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{field} must be a name containing letters, digits, underscores, dots or hyphens")
    return value


def merge_config(base, overrides):
    """Merge nested config mappings without mutating either input."""
    result = deepcopy(base)
    if "criterion" in overrides and overrides["criterion"] != base.get("criterion"):
        result.pop("criterion_kwargs", None)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_experiment_config(path, _parents=()):
    """Resolve relative YAML extends (left to right, then the child)."""
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError(f"cyclic config inheritance: {path}")
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    parents = config.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(not isinstance(parent, str) for parent in parents):
        raise ValueError(f"extends must be a config path or list of paths: {path}")
    resolved = {}
    for parent in parents:
        resolved = merge_config(resolved, load_experiment_config(path.parent / parent, (*_parents, path)))
    return merge_config(resolved, config)


def phase1_config(config):
    """Name the shared encoder independently of its phase-two child."""
    name = validate_experiment_name(config["phase1_experiment"], "phase1_experiment")
    return {**config, "framework": "lewm_encoder", "experiment_name": name}


def model_path(root_dir, config, fname, version):
    tag = config.get("tag_jepa")
    return os.path.join(experiment_base_dir(root_dir, config, fname, version),
                        f"jepa_{tag}" if tag else "jepa", "model.pth.tar")


def entry_overrides(args, updates=None):
    """CLI scoring selects an action without changing the experiment config."""
    overrides = dict(updates or {})
    for key in ("stage", "pretrained_from", "score_checkpoint", "phase1_version"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if getattr(args, "score", False):
        overrides["stage"] = "score"
    return overrides


def dataset_output_dir(root_dir, config):
    """Optional experiment group directly below the dataset output directory."""
    base = os.path.join(root_dir, config.get("dataset_collection", config["train_db_name"]))
    group = config.get("experiment_version")
    if group is not None:
        base = os.path.join(base, validate_experiment_name(group, "experiment_version"))
    return base


def experiment_base_dir(root_dir, config, fname, version):
    """Tags identify stable paths; explicit named versions add a run directory."""
    fname = config.get("output_series", fname)
    version = DEFAULT_VERSION if version is None else version
    validate_experiment_name(version, "version")
    dataset_dir = dataset_output_dir(root_dir, config)
    run_parts = [] if version == DEFAULT_VERSION else [version]
    # Single-series datasets need no repeated dataset identifier. Machine
    # datasets retain their file component to isolate weights and reports.
    series_parts = [] if fname == config["train_db_name"] else [fname]
    if config.get("framework"):
        framework = validate_experiment_name(config["framework"], "framework")
        experiment = validate_experiment_name(config.get("experiment_name", "default"), "experiment_name")
        validate_experiment_name(version, "version")
        validate_experiment_name(fname, "fname")
        if config.get("phase1_experiment") is not None or framework == "lewm_encoder":
            encoder = validate_experiment_name(
                config.get("phase1_experiment", experiment), "phase1_experiment")
            if framework in ("lewm", "lewm_encoder"):
                branch = ["phase1"]
            elif framework in ("reconstruction", "lewm_reconstruction"):
                branch = ["reconstruction", experiment]
            elif framework in ("cross_attention", "lewm_cross_attention"):
                branch = ["cross_attention", experiment]
            else:
                raise ValueError(f"framework {framework} cannot use phase1_experiment")
            return os.path.join(dataset_dir, "lewm_encoder",
                                encoder, *branch, *run_parts, *series_parts)
        return os.path.join(dataset_dir, framework, experiment, *run_parts, *series_parts)
    return os.path.join(dataset_dir, *run_parts, *series_parts)


def _training_sections(config):
    """Map grouped training YAML onto the existing runtime settings."""
    if "train_hyperparameters" not in config and "data" not in config:
        return config
    p = dict(config)
    framework = p.get("framework")
    if framework not in ("ae", "vae", "lewm"):
        raise ValueError("framework must be ae, vae, or lewm")

    data = dict(p.pop("data", {}))
    for source, target in (("name", "train_db_name"), ("file", "fname"),
                           ("root", "dataset_root"), ("window", "wsz")):
        if source in data:
            data[target] = data.pop(source)
    p.update(data)
    for key in ("train_db_name", "wsz", "stride"):
        if key not in p:
            raise ValueError(f"data requires {key}")
    p.setdefault("val_db_name", p["train_db_name"])

    training = dict(p.pop("train_hyperparameters", {}))
    for kind, default in (("optimizer", "adamw"), ("scheduler", "constant")):
        options = training.pop(kind, {"name": default})
        if isinstance(options, str):
            options = {"name": options}
        options = dict(options)
        training[kind] = options.pop("name", default)
        training[f"{kind}_kwargs"] = options
    training["optimizer_kwargs"].setdefault("lr", 1e-3)
    p.update(training)
    for key, default in (("setup", "jepa"), ("batch_size", 32), ("num_workers", 0),
                         ("epochs", 1)):
        p.setdefault(key, default)

    loss = dict(p.pop("loss", {}))
    default_loss = "lewm" if framework == "lewm" else "reconstruction"
    p["criterion"] = loss.pop("name", default_loss)
    if p["criterion"] == "reconstruction":
        loss.setdefault("beta", 0.3 if framework == "vae" else 0.0)
    p["criterion_kwargs"] = loss

    scorer = dict(p.pop("scorer", {}))
    p["score_mode"] = scorer.pop("mode", "l1")
    p["score_tau"] = scorer.pop("tau", 1.0)
    calibration = dict(p.get("calibration_kwargs", {}))
    quantile = calibration.get("quantile", p.get("calibration_quantile", 0.99))
    p["calibration_quantile"] = scorer.pop("calibration_quantile", quantile)
    if calibration:
        p["calibration_kwargs"] = {**calibration, "quantile": p["calibration_quantile"]}
    if framework == "lewm":
        modules = {role: p.pop(role) for role in
                   ("engine", "backbone", "encoder", "action", "predictor", "head") if role in p}
        if "domain" in scorer:
            modules["score_domain"] = scorer.pop("domain")
        if "name" in scorer:
            modules["score_fn"] = scorer.pop("name")
        p["lewm_modules"] = modules
    else:
        encoder = dict(p.pop("encoder", {}))
        feature_backbone = p.pop("backbone", None)
        p["feature_modules"] = {
            "encoder": dict(encoder), "backbone": feature_backbone,
            "freeze_encoder": bool(p.get("freeze_encoder", encoder.get("freeze", False)))}
        p["backbone"] = encoder.pop("name", "resnet")
        p["model_kwargs"] = {k: v for k, v in encoder.items() if k not in ("freeze", "pretrained_from", "source")}
        head = dict(p.pop("head", {}) or {})
        if head.pop("name", "mirrored") != "mirrored":
            raise ValueError("AE and VAE use the mirrored reconstruction head")
        p["recon_kwargs"] = head
        p["arm"] = framework
        if scorer.pop("domain", "raw") != "raw" or scorer.pop("name", "residual") != "residual":
            raise ValueError("AE and VAE use raw residual scores")
    if scorer:
        raise ValueError(f"unknown scorer options: {', '.join(scorer)}")
    return p


def create_config(config_file_env, config_file_exp, fname, version=None, update_dictionary=None,
                  *, output_dir=None):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    config = _training_sections(load_experiment_config(config_file_exp))
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    for k, v in (update_dictionary or {}).items():
        cfg[k] = v
    fname = fname or cfg.get("fname") or cfg["train_db_name"]
    
    # Set paths for pretext task (These directories are needed in every stage)
    version = DEFAULT_VERSION if version is None else version
    base_dir = str(output_dir) if output_dir is not None else experiment_base_dir(root_dir, cfg, fname, version)
    if cfg.get("phase1_experiment") and cfg.get("framework") in (
            "reconstruction", "lewm_reconstruction", "cross_attention", "lewm_cross_attention"):
        cfg["phase1_version"] = cfg.get("phase1_version") or version
        # The inherited phase-one tag is independent of each child's tag.
        source_config = phase1_config(cfg)
        source_config["tag_jepa"] = cfg.get("tag_phase1", cfg.get("tag_jepa"))
        cfg["phase1_checkpoint"] = model_path(root_dir, source_config, fname,
                                             cfg["phase1_version"])
        if not cfg.get("pretrained_from"):
            cfg["pretrained_from"] = cfg["phase1_checkpoint"]
    
    pretext_tag = cfg.get('tag_pretext', None)
    cfg['pretext_tag'] = ("_"+pretext_tag) if pretext_tag else ""
    pretext_dir = os.path.join(base_dir, f'pretext{cfg['pretext_tag']}')
    mkdir(base_dir)
    cfg['version'] = version
    cfg['root_dir'] = root_dir
    cfg['experiment_dir'] = base_dir
    cfg['pretext_dir'] = pretext_dir
    cfg['fname'] = fname

    jepa_tag = cfg.get('tag_jepa', None)
    cfg['jepa_tag'] = ("_"+jepa_tag) if jepa_tag else ""
    jepa_dir = base_dir if output_dir is not None else os.path.join(base_dir, f'jepa{cfg['jepa_tag']}')
    mkdir(base_dir)
    mkdir(jepa_dir)
    cfg['jepa_dir'] = jepa_dir
    cfg['jepa_checkpoint'] = os.path.join(jepa_dir, 'checkpoint.pth.tar')
    cfg['jepa_model'] = os.path.join(jepa_dir, 'model.pth.tar')
    cfg['jepa_model_best'] = os.path.join(jepa_dir, "model_best_eval.pth.tar")
    cfg['calibration_path'] = os.path.join(jepa_dir, 'calibration.json')
    cfg['scores_path'] = os.path.join(jepa_dir, 'scores.npz')
    cfg['metrics_path'] = os.path.join(jepa_dir, 'metrics.json')
        

    if "res_kwargs" in cfg:
        cfg["res_kwargs"]["window_size"] = cfg["wsz"]
    return cfg
