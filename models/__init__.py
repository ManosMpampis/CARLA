"""Backbone registry for the steered anomaly detector."""

__all__ = ["BACKBONE_REGISTRY", "get_backbone"]


def _build_steered_resnet(**kwargs):
    from models.steered_lewm import SteeredResNetEncoder

    encoder = SteeredResNetEncoder(**kwargs)
    return {"model": encoder, "dim": [encoder.output_dims]}


BACKBONE_REGISTRY = {
    "steered_resnet": _build_steered_resnet,
}


def get_backbone(name, **kwargs):
    """Build a registered backbone by config name."""
    if name not in BACKBONE_REGISTRY:
        raise ValueError("Invalid backbone {}".format(name))
    return BACKBONE_REGISTRY[name](**kwargs)
