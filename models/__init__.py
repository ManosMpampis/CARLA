"""Model registry: the single wiring point for backbone construction.

Every model family registers a factory keyed by its config name
(`backbone:` in experiment YAMLs). Adding an arm means registering a
builder here — never copying driver scripts.
"""
from models.models import ContrastiveModel, ClusteringModel
from models.resnet_ts import resnet_ts

__all__ = [
    "resnet_ts",
    "ContrastiveModel",
    "ClusteringModel",
    "BACKBONE_REGISTRY",
    "get_backbone",
]


def _build_resnet_ts(**kwargs):
    return resnet_ts(**kwargs)


def _build_jepa_pyramid(**kwargs):
    from models.jepa_pyramid import PyramidEncoder

    encoder = PyramidEncoder(**kwargs)
    return {"model": encoder, "dim": encoder.level_dims}


def _build_jepa_transformer(**kwargs):
    from models.jepa_transformer import JEPATransformer

    transformer = JEPATransformer(**kwargs)
    return {"model": transformer, "dim": transformer.level_dims}


BACKBONE_REGISTRY = {
    "resnet_ts": _build_resnet_ts,
    "jepa_pyramid": _build_jepa_pyramid,
    "jepa_transformer": _build_jepa_transformer,
}


def get_backbone(name, **kwargs):
    """Build a registered backbone by config name.

    Legacy backbones (resnet_ts) keep returning the historical
    {'backbone': ..., 'dim': ...} mapping; JEPA arms return
    {'model': ..., 'dim': ...} where `model` exposes encode/predict.
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError("Invalid backbone {}".format(name))
    return BACKBONE_REGISTRY[name](**kwargs)
