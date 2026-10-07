"""Backbone registry for LeWM experiments."""

__all__ = ["BACKBONE_REGISTRY", "get_backbone"]


def _build_lewm_resnet(**kwargs):
    from models.lewm import LeWMResNetEncoder

    encoder = LeWMResNetEncoder(**kwargs)
    return {"model": encoder, "dim": [encoder.output_dims]}


BACKBONE_REGISTRY = {
    "lewm_resnet": _build_lewm_resnet,
}


def get_backbone(name, **kwargs):
    """Build a registered backbone by config name."""
    if name not in BACKBONE_REGISTRY:
        raise ValueError("Invalid backbone {}".format(name))
    return BACKBONE_REGISTRY[name](**kwargs)
