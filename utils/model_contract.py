"""Persist feature functions and task semantics for exact transfer and reload.

The runtime prepares a config before model construction, initializes sources
only for fresh runs, and saves the prepared config in every destination snapshot.
An embedded architecture makes destination resume independent of source files.
"""

from copy import deepcopy
import hashlib
import inspect
from pathlib import Path

import torch
import numpy as np


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


class ModelContract:
    """Reject equal-shaped checkpoints representing a different experiment."""

    def get_extra_state(self):
        return deepcopy(getattr(self, "_model_contract", {}))

    def set_extra_state(self, state):
        expected = getattr(self, "_model_contract", {})
        if state != expected:
            raise ValueError("checkpoint model/task/feature schema differs from configured experiment")


def architecture(name, cls, options):
    """Canonical constructor defaults, including non-tensor architecture choices."""
    signature = inspect.signature(cls.__init__)
    result = {key: deepcopy(value.default) for key, value in signature.parameters.items()
              if key != "self" and value.default is not inspect.Parameter.empty
              and value.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)}
    result.update(deepcopy(options))
    # TCNEncoder's variadic constructor inherits Encoder's options.
    if name == "tcn":
        from models.encoders import Encoder
        inherited = architecture("convolutional", Encoder, options)["options"]
        inherited.setdefault("norm", "none")
        result = {**inherited, **result}
    return {"name": name, "options": _plain(result)}


def effective_architecture(model):
    modules = [getattr(model, "backbone", None), model.encoder]
    parts = [deepcopy(module._feature_architecture) for module in modules
             if module is not None and module._feature_architecture["name"] != "identity"]
    return parts[0] if len(parts) == 1 else {"name": "composed", "options": {"parts": parts}}


def attach_model_contract(model, config):
    """Record actual feature modules and the complete prediction/scoring task."""
    feature = effective_architecture(model)
    roles = {role: deepcopy(getattr(model, role)._feature_architecture)
             for role in ("backbone", "encoder") if getattr(model, role, None) is not None}
    if config.get("framework") == "lewm":
        task = deepcopy(config.get("lewm_modules", {}))
        for role in ("backbone", "encoder"):
            task[role] = roles.get(role)
        task["attention_layout"] = "future_query_history_key_value_v2"
        task["spectral_layout"] = "grouped_complex_channels_v2"
        task["frozen_roles"] = sorted(model._frozen | ({"encoder"} if model.encoder_frozen else set()))
    else:
        task = {"features": roles, "variational": bool(model.variational),
                "decoder": deepcopy(config.get("recon_kwargs", {})),
                "freeze_encoder": bool(model.encoder_frozen),
                "freeze_backbone": bool(getattr(model, "backbone_frozen", False))}
    # Initialization provenance is not part of a module's feature function.
    for role in ("engine", "action", "predictor", "head"):
        if isinstance(task.get(role), dict):
            task[role].pop("pretrained_from", None)
    model._model_contract = {
        "version": 2, "framework": config.get("framework"),
        "feature_schema": list(config.get("_feature_schema", [])),
        "feature_architectures": {**roles, "effective": feature},
        "task": _plain(task), "score_mode": config.get("score_mode", "l1"),
        "score_tau": config.get("score_tau", 1.0),
        "latent_objective": deepcopy(config.get("criterion_kwargs", {})),
    }
    model.transfer_provenance = deepcopy(config.get("_feature_transfer", {}))
    return model


def _roles(config):
    return config["lewm_modules"] if config.get("framework") == "lewm" else config.setdefault("feature_modules", {})


def _checkpoint_contract(checkpoint, destination_config=None, feature_schema=None):
    state = checkpoint.get("model", checkpoint.get("state_dict", {}))
    contract = checkpoint.get("model_contract") or state.get("_extra_state")
    if not contract or contract.get("version") != 2:
        return _legacy_source_contract(checkpoint, destination_config, feature_schema)
    return contract


def _legacy_source_contract(checkpoint, destination_config, feature_schema):
    """Recover old feature constructors without reinterpreting their activations."""
    from models import ENCODER_REGISTRY, build_feature_module

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("legacy source checkpoint requires its saved constructor config")
    modules = config.get("lewm_modules")
    if not modules:
        modules = {"encoder": {"name": config.get("backbone", "lewm_resnet"),
                                **config.get("model_kwargs", {})}}
    descriptors = {}
    state = checkpoint.get("model", checkpoint.get("state_dict", {}))
    for role in ("backbone", "encoder"):
        options = deepcopy(modules.get(role) or {})
        name = options.pop("name", "none" if role == "backbone" else "resnet")
        if name == "none":
            continue
        for key in ("freeze", "pretrained_from", "checkpoint_prefix"):
            options.pop(key, None)
        input_key = "in_channels" if name in ("resnet", "lewm_resnet") else "in_dim"
        if input_key not in options:
            first_weight = next((value for key, value in state.items()
                                 if key.startswith(role + ".") and key.endswith("weight")
                                 and torch.is_tensor(value) and value.ndim == 3), None)
            if first_weight is not None:
                options[input_key] = int(first_weight.shape[1])
        if name in ("resnet", "lewm_resnet"):
            options.setdefault("latent_output", "legacy")
        descriptor = architecture(name, ENCODER_REGISTRY[name], options)
        build_feature_module(descriptor)  # Validate the complete saved constructor.
        descriptors[role] = descriptor
    parts = [descriptor for descriptor in descriptors.values() if descriptor["name"] != "identity"]
    effective = parts[0] if len(parts) == 1 else {"name": "composed", "options": {"parts": parts}}
    source_schema = (checkpoint.get("feature_schema") or config.get("_feature_schema")
                     or checkpoint.get("inference", {}).get("feature_schema"))
    basis = "saved_feature_schema"
    if not source_schema:
        source_collection = config.get("dataset_collection", config.get("train_db_name"))
        target_collection = (destination_config or {}).get("dataset_collection", (destination_config or {}).get("train_db_name"))
        source_file, target_file = config.get("fname"), (destination_config or {}).get("fname")
        raw_width = build_feature_module(effective).input_dims
        if (source_collection != target_collection or source_file != target_file
                or feature_schema is None or len(feature_schema) != raw_width):
            raise ValueError("legacy source has no sensor schema; require matching saved collection, series and input width")
        source_schema = list(feature_schema)
        basis = "saved_collection_series_width_with_destination_reader_order"
    return {"version": 2, "source_format": "legacy_constructor_config", "schema_validation": basis,
            "framework": config.get("framework", config.get("arm")),
            "feature_schema": list(source_schema),
            "feature_architectures": {**descriptors, "effective": effective},
            "task": {"legacy_source_feature_function_only": True}}


def _validate_source_options(options, descriptor):
    explicit = {key: value for key, value in options.items()
                if key not in ("name", "pretrained_from", "checkpoint_prefix", "freeze", "architecture")}
    if explicit and any(descriptor["options"].get(key) != _plain(value) for key, value in explicit.items()):
        raise ValueError("requested source architecture differs from checkpoint feature function")
    if options.get("name", "pretrained") not in ("pretrained", descriptor["name"]):
        raise ValueError("requested source encoder type differs from checkpoint")


def _normalization_changed(source, destination):
    """Record scaler changes without replacing honest destination normalization."""
    if not source or not destination:
        return None
    if "members" in source or "members" in destination:
        if set(source.get("members", {})) != set(destination.get("members", {})):
            return True
        return any(_normalization_changed(source["members"][name], destination["members"][name])
                   for name in source["members"])
    changed = False
    for name in ("mean", "std"):
        actual = source.get(name, source.get(f"normalization_{name}"))
        expected = destination.get(name, destination.get(f"normalization_{name}"))
        if actual is not None and expected is not None:
            if np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, rtol=1e-8, atol=1e-10):
                changed = True
    if source.get("feature_schema") and destination.get("feature_schema"):
        if source["feature_schema"] != destination["feature_schema"]:
            raise ValueError("source preprocessing feature schema/order differs")
    return changed


def prepare_feature_source_config(config, feature_schema, resume_checkpoint=None):
    """Resolve source architecture before building, without loading destination weights."""
    result = deepcopy(dict(config))
    result["_feature_schema"] = list(feature_schema)
    roles = _roles(result)
    saved_roles = _roles(deepcopy(resume_checkpoint["config"])) if resume_checkpoint is not None else {}
    provenance = deepcopy(result.get("_feature_transfer", {}))
    for role in ("backbone", "encoder"):
        options = roles.get(role)
        if not isinstance(options, dict) or not options.get("pretrained_from"):
            continue
        options = deepcopy(options)
        saved = saved_roles.get(role, {})
        if resume_checkpoint is not None:
            if not saved.get("architecture"):
                raise ValueError("destination checkpoint has no embedded source architecture")
            # Retain caller's freeze choice for contract validation at load.
            options["architecture"] = deepcopy(saved["architecture"])
            _validate_source_options(options, options["architecture"])
            provenance = deepcopy(resume_checkpoint["config"].get("_feature_transfer", {}))
        else:
            source_path = Path(options["pretrained_from"])
            checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
            contract = _checkpoint_contract(checkpoint, result, feature_schema)
            source_preprocessing = checkpoint.get("preprocessing", checkpoint.get("inference", {}))
            changed = _normalization_changed(source_preprocessing, result.get("_preprocessing"))
            source_schema = contract.get("feature_schema") or checkpoint.get("feature_schema")
            if list(source_schema or []) != list(feature_schema):
                raise ValueError("source checkpoint feature schema/order must match destination")
            descriptor = contract["feature_architectures"]["effective"]
            if "architecture" in options and options["architecture"] != descriptor:
                raise ValueError("embedded source architecture differs from source checkpoint")
            _validate_source_options(options, descriptor)
            options["architecture"] = descriptor
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            provenance[role] = {"checkpoint": str(source_path.resolve()), "sha256": digest,
                                "selection": checkpoint.get("selection"),
                                "source_contract": contract,
                                "preprocessing": deepcopy(source_preprocessing),
                                "destination_preprocessing": deepcopy(result.get("_preprocessing")),
                                "normalization_changed": changed}
        options = {key: value for key, value in options.items()
                   if key in ("pretrained_from", "checkpoint_prefix", "freeze", "architecture")}
        options["name"] = "pretrained"
        roles[role] = options
    result["_feature_transfer"] = provenance
    return result


def initialize_feature_source(model, config, feature_schema=None):
    """Copy complete effective source features into requested destination roles."""
    for role, source in config.get("_feature_transfer", {}).items():
        path = Path(source["checkpoint"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("source checkpoint changed after preflight")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        contract = _checkpoint_contract(checkpoint, config, feature_schema)
        if feature_schema is not None and list(contract["feature_schema"]) != list(feature_schema):
            raise ValueError("source feature schema/order differs")
        source_roles = [name for name in ("backbone", "encoder")
                        if name in contract["feature_architectures"]
                        and contract["feature_architectures"][name]["name"] != "identity"]
        source_state = checkpoint.get("model", checkpoint.get("state_dict", {}))
        state = {}
        for index, source_role in enumerate(source_roles):
            prefix = source_role + "."
            destination_prefix = "" if len(source_roles) == 1 else f"{index}."
            state.update({destination_prefix + key.removeprefix(prefix): value
                          for key, value in source_state.items() if key.startswith(prefix)})
        if not state:
            raise ValueError("source checkpoint contains no encoder feature weights")
        destination = getattr(model, role)
        destination.load_state_dict(state, strict=True)
    return model
