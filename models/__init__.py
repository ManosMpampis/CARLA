"""Model registry: the single wiring point for backbone construction.

Every family registers a factory keyed by config name (`backbone:` in
experiment YAMLs). Adding an arm means registering a builder here.
"""
from models.models import ContrastiveModel, ClusteringModel
from models.resnet_ts import resnet_ts

__all__ = ["resnet_ts", "ContrastiveModel", "ClusteringModel",
           "BACKBONE_REGISTRY", "get_backbone"]


def _build_resnet_ts(**kwargs):
    return resnet_ts(**kwargs)


def _build_jepa_pyramid(**kwargs):
    from models.encoder import PyramidEncoder

    encoder = PyramidEncoder(**kwargs)
    return {"model": encoder, "dim": encoder.level_dims}


def _build_jepa_transformer(**kwargs):
    from models.encoder import TransformerEncoder1d

    enc = TransformerEncoder1d(**kwargs)
    return {"model": enc, "dim": enc.level_dims}


def _build_fourier_pyramid(**kwargs):
    from models.fourier_pyramid import FourierPyramidEncoder

    encoder = FourierPyramidEncoder(**kwargs)
    return {"model": encoder, "dim": encoder.level_dims}


def _build_tfscout_stub(**kwargs):
    """Placeholder trunk entry (ticket 13): validates config plumbing.

    Full frequency pathway + steering + FPN fusion lands behind this key
    in ticket order; until then it builds the time pyramid so the head
    tournament harness has a stable target.
    """
    from models.encoder import PyramidEncoder

    encoder = PyramidEncoder(**kwargs)
    return {"model": encoder, "dim": encoder.level_dims}


def _build_steered_resnet(**kwargs):
    from models.steered_lewm import SteeredResNetEncoder

    encoder = SteeredResNetEncoder(**kwargs)
    return {"model": encoder, "dim": [encoder.output_dims]}


BACKBONE_REGISTRY = {
    "resnet_ts": _build_resnet_ts,
    "steered_resnet": _build_steered_resnet,
    "jepa_pyramid": _build_jepa_pyramid,
    "jepa_transformer": _build_jepa_transformer,
    "fourier_pyramid": _build_fourier_pyramid,
    "tfscout": _build_tfscout_stub,
}


def get_backbone(name, **kwargs):
    """Build a registered backbone by config name."""
    if name not in BACKBONE_REGISTRY:
        raise ValueError("Invalid backbone {}".format(name))
    return BACKBONE_REGISTRY[name](**kwargs)
