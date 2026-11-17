"""Backbone registry for LeWM experiments."""

__all__ = ["BACKBONE_REGISTRY", "get_backbone", "FRAMEWORK_REGISTRY", "get_framework"]


def _build_encoder_resnet(**kwargs):
    from models.encoders import ResNetEncoder

    encoder = ResNetEncoder(**kwargs)
    return {"model": encoder, "dim": [encoder.output_dims]}


ENCODER_REGISTRY = {
    "resnet": _build_encoder_resnet,
}


def get_encoder(name, **kwargs):
    """Build a registered backbone by config name."""
    if name not in ENCODER_REGISTRY:
        raise ValueError("Invalid backbone {}".format(name))
    return ENCODER_REGISTRY[name](**kwargs)


def _build_guided_forecasting(encoder, **kwargs):
    from models.lw_guided_forecasting import CrossAttentionLeWM

    return CrossAttentionLeWM(encoder, **kwargs)


def _build_reconstruction(encoder, **kwargs):
    from models.lw_reconstructor import LeWMModel

    return LeWMModel(encoder, **kwargs)


# def _build_reconstruction(encoder, **kwargs):
#     from models.recon_model import ReconModel

#     return ReconModel(encoder, **kwargs)


def _build_ae(encoder, **kwargs):
    from models.ae_baseline import ReconstructionBaseline

    return ReconstructionBaseline(encoder=encoder, variational=False, **kwargs)


def _build_vae(encoder, **kwargs):
    from models.ae_baseline import ReconstructionBaseline

    return ReconstructionBaseline(encoder=encoder, variational=True, **kwargs)


FRAMEWORK_REGISTRY = {
    # "lewm_": _build_lewm,
    "lewm_reconstruction": _build_reconstruction,
    "lewm_guided_forecasting": _build_guided_forecasting,
    "ae": _build_ae,
    "vae": _build_vae,
}


def get_framework(name, encoder, **kwargs):
    """Build a registered framework around a registered backbone."""
    if name not in FRAMEWORK_REGISTRY:
        raise ValueError(f"Invalid framework {name}")
    return FRAMEWORK_REGISTRY[name](encoder, **kwargs)
