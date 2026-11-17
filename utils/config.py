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

def model_path(root_dir, config, fname, version):
    tag = config.get("tag", "")
    return os.path.join(experiment_base_dir(root_dir, config, fname, version), f"{tag}", "model.pth.tar")

def dataset_output_dir(root_dir, config):
    """Optional experiment group directly below the dataset output directory."""
    base = os.path.join(root_dir, config["train_db_name"])
    group = config.get("experiment_version")
    if group is not None:
        base = os.path.join(base, validate_experiment_name(group, "experiment_version"))
    return base

def experiment_base_dir(root_dir, config, fname, version):
    """Tags identify stable paths; explicit named versions add a run directory."""
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


def create_config(config_file_env, config_file_exp, fname, version=None, update_dictionary=None,
                  *, output_dir=None):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']
   
    config = load_experiment_config(config_file_exp)
    
    cfg = EasyDict()
   
    # Copy
    for k, v in config.items():
        cfg[k] = v

    for k, v in (update_dictionary or {}).items():
        cfg[k] = v

    cfg["framework"]["amp"] = cfg.get("amp")
    fname = fname or cfg.get("fname") or cfg["train_db_name"]
    
    # Set paths for pretext task (These directories are needed in every stage)
    version = DEFAULT_VERSION if version is None else version
    base_dir = str(output_dir) if output_dir is not None else experiment_base_dir(root_dir, cfg, fname, version)
    
    mkdir(base_dir)
    cfg['version'] = version
    cfg['experiment_dir'] = base_dir
    cfg['fname'] = fname

    tag = cfg.get('tag_jepa', None)
    cfg['tag'] = ("_"+tag) if tag else ""
    tag_dir = base_dir if output_dir is not None else os.path.join(base_dir, cfg['tag'])
    mkdir(tag_dir)
    cfg['tag_dir'] = tag_dir
    cfg['checkpoint'] = os.path.join(cfg['tag_dir'], 'checkpoint.pth.tar')
    cfg['model'] = os.path.join(cfg['tag_dir'], 'model.pth.tar')
    cfg['model_best'] = os.path.join(cfg['tag_dir'], "model_best_eval.pth.tar")
    cfg['calibration_path'] = os.path.join(cfg['tag_dir'], 'calibration.json')
    cfg['scores_path'] = os.path.join(cfg['tag_dir'], 'scores.npz')
    cfg['metrics_path'] = os.path.join(cfg['tag_dir'], 'metrics.json')
        
    cfg["framework"]["ecoder"]["window_size"] = cfg["wsz"]
    return cfg
