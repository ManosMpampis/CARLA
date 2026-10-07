import os
import time
from copy import deepcopy
from pathlib import Path
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing as mkdir


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


def experiment_base_dir(root_dir, config, fname, version):
    """Resolve named framework/experiment runs while preserving legacy paths."""
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
            return os.path.join(root_dir, config["train_db_name"], "lewm_encoder",
                                encoder, *branch, version, fname)
        return os.path.join(root_dir, config["train_db_name"], framework, experiment, version, fname)
    return os.path.join(root_dir, config["train_db_name"], version, fname)


def create_config(config_file_env, config_file_exp, fname, version=None, update_dictionary=None):
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
    
    # Set paths for pretext task (These directories are needed in every stage)
    version = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) if version is None else version
    base_dir = experiment_base_dir(root_dir, cfg, fname, version)
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
    cfg['experiment_dir'] = base_dir
    cfg['pretext_dir'] = pretext_dir
    cfg['fname'] = fname

    if cfg['setup'] == 'jepa':
        jepa_tag = cfg.get('tag_jepa', None)
        cfg['jepa_tag'] = ("_"+jepa_tag) if jepa_tag else ""
        jepa_dir = os.path.join(base_dir, f'jepa{cfg['jepa_tag']}')
        mkdir(base_dir)
        mkdir(jepa_dir)
        cfg['jepa_dir'] = jepa_dir
        cfg['jepa_checkpoint'] = os.path.join(jepa_dir, 'checkpoint.pth.tar')
        cfg['jepa_model'] = os.path.join(jepa_dir, 'model.pth.tar')
        cfg['jepa_model_best'] = os.path.join(jepa_dir, "model_best_eval.pth.tar")
        cfg['calibration_path'] = os.path.join(jepa_dir, 'calibration.json')
        cfg['scores_path'] = os.path.join(jepa_dir, 'scores.npz')
        cfg['metrics_path'] = os.path.join(jepa_dir, 'metrics.json')

    if cfg['setup'] in ['classification', 'classification_e2e']:
        classification_tag = cfg.get('tag_class', None)
        cfg['classification_tag'] = ("_"+classification_tag) if classification_tag else ""
        classification_dir = os.path.join(base_dir, f'classification{cfg['classification_tag']}')
        mkdir(base_dir)
        mkdir(classification_dir)
        cfg['classification_dir'] = classification_dir
        cfg['classification_checkpoint'] = os.path.join(classification_dir, 'checkpoint.pth.tar')
        cfg['classification_checkpoint_last'] = os.path.join(classification_dir, 'checkpoint_last.pth.tar')
        cfg['classification_model'] = os.path.join(classification_dir, 'model.pth.tar')
        cfg['classification_trainfeatures'] = os.path.join(classification_dir, 'classification_traintfeatures.csv')
        cfg['classification_trainprobs'] = os.path.join(classification_dir, 'classification_trainprobs.csv')
        cfg['classification_testfeatures'] = os.path.join(classification_dir, 'classification_testtfeatures.csv')
        cfg['classification_testprobs'] = os.path.join(classification_dir, 'classification_testprobs.csv')
        # Evaluation paths
        mkdir(os.path.join(classification_dir, 'best'))
        cfg['eval_train_csl'] = os.path.join(classification_dir, 'best', 'eval_train_cls.csv')
        cfg['eval_train_best'] = os.path.join(classification_dir, 'best', 'eval_train_best.csv')
        cfg['eval_test_cls'] = os.path.join(classification_dir, 'best', 'eval_test_cls.csv')
        cfg['eval_test_best'] = os.path.join(classification_dir, 'best', 'eval_test_best.csv')
        cfg['eval_test_train_th'] = os.path.join(classification_dir, 'best', 'eval_test_train_th.csv')
        cfg['eval_tstest_cls'] = os.path.join(classification_dir, 'best', 'eval_timeseries_cls.csv')
        cfg['eval_tstest_best'] = os.path.join(classification_dir, 'best', 'eval_timeseries_best.csv')
        cfg['eval_tstest_trainth'] = os.path.join(classification_dir, 'best', 'eval_timeseries_train_th.csv')
        mkdir(os.path.join(classification_dir, 'cls'))
        cfg['clseval_train_csl'] = os.path.join(classification_dir, 'cls', 'eval_train_cls.csv')
        cfg['clseval_train_best'] = os.path.join(classification_dir, 'cls', 'eval_train_best.csv')
        cfg['clseval_test_cls'] = os.path.join(classification_dir, 'cls', 'eval_test_cls.csv')
        cfg['clseval_test_best'] = os.path.join(classification_dir, 'cls', 'eval_test_best.csv')
        cfg['clseval_test_train_th'] = os.path.join(classification_dir, 'cls', 'eval_test_train_th.csv')
        cfg['clseval_tstest_cls'] = os.path.join(classification_dir, 'cls', 'eval_timeseries_cls.csv')
        cfg['clseval_tstest_best'] = os.path.join(classification_dir, 'cls', 'eval_timeseries_best.csv')
        cfg['clseval_tstest_trainth'] = os.path.join(classification_dir, 'cls', 'eval_timeseries_train_th.csv')
        

    if "res_kwargs" in cfg:
        cfg["res_kwargs"]["window_size"] = cfg["wsz"]
    return cfg
