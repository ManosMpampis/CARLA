"""Backbone registry for LeWM experiments."""

__all__ = ["BACKBONE_REGISTRY", "get_backbone", "FRAMEWORK_REGISTRY", "get_framework"]


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


def _build_cross_attention(encoder, **kwargs):
    from models.lewm_cross_attention import CrossAttentionLeWM

    return CrossAttentionLeWM(encoder, **kwargs)


def _build_lewm(encoder, **kwargs):
    from models.lewm import LeWMModel

    return LeWMModel(encoder, **kwargs)


def _build_reconstruction(encoder, **kwargs):
    from models.recon_model import ReconModel

    return ReconModel(encoder, **kwargs)


def _build_ae(encoder, **kwargs):
    from models.ae_baseline import ReconstructionBaseline

    return ReconstructionBaseline(encoder=encoder, variational=False, **kwargs)


def _build_vae(encoder, **kwargs):
    from models.ae_baseline import ReconstructionBaseline

    return ReconstructionBaseline(encoder=encoder, variational=True, **kwargs)


FRAMEWORK_REGISTRY = {
    "lewm": _build_lewm,
    "lewm_reconstruction": _build_reconstruction,
    "lewm_cross_attention": _build_cross_attention,
    "ae": _build_ae,
    "vae": _build_vae,
}


def get_framework(name, encoder, **kwargs):
    """Build a registered framework around a registered backbone."""
    if name not in FRAMEWORK_REGISTRY:
        raise ValueError(f"Invalid framework {name}")
    return FRAMEWORK_REGISTRY[name](encoder, **kwargs)
