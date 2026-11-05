"""Backbone registry for LeWM experiments."""

from models.input_engines import ForecastEngine, MaskedEngine
from models.encoders import ResNetEncoder
from models.actions import MaskedRepresentationAction, MaskedRepresentationAuxiliaryGuidedAction, TCNEncodedAction
from models.ae_baseline import ReconstructionBaseline
from models.lewm import LeWM
from models.blocks import Identity
from utils.reconstruction_scores import reconstruction_score_map

__all__ = ["decoders", "encoders", "actions", "get_framework"]

def _build_cross_attention(encoder, **kwargs):
    from models.lewm_cross_attention import CrossAttentionLeWM

    return CrossAttentionLeWM(encoder, **kwargs)


def _build_cross_attention_tcn(encoder, **kwargs):
    from models.lewm_cross_attention_tcn import CrossAttentionTCNLeWM

    return CrossAttentionTCNLeWM(encoder, **kwargs)


def _build_rec_ca(encoder, **kwargs):
    from models.lewm_rec_ca import RecCALeWM

    return RecCALeWM(encoder, **kwargs)


def _build_reconstruction(encoder, **kwargs):
    from models.recon_model import ReconModel

    return ReconModel(encoder, **kwargs)

FRAMEWORK_REGISTRY = {
    "lewm": LeWM,
    "lewm_reconstruction": _build_reconstruction,
    "lewm_cross_attention": _build_cross_attention,
    "lewm_cross_attention_tcn": _build_cross_attention_tcn,
    "lewm_rec_ca": _build_rec_ca,
    "ae": ReconstructionBaseline,
    "vae": ReconstructionBaseline,
}

ENCODER_REGISTRY = {
    "resnet": ResNetEncoder,
    "none": Identity,
}

ACTION_REGISTRY = {
    "masked": "MaskedAction",
    "tcn": "TCNAction",
    "convolutional": "ConvolutionalAction",
    "auxilary": "AuxiliaryAction",
    "none": Identity,
}

DECODER_REGISTRY = {
    "fft": "MirroredDecoder",
    "convolutional": "ConvolutionalDecoder",
    "none": Identity,
}

INPUT_ENGINE_REGISTRY = {
    "mask": MaskedEngine,
    "forecast": ForecastEngine,
    "none": Identity,
}

SCORE_FUNCTIONS = {
    "reconstruction": reconstruction_score_map,
}

def get_framework(name, **kwargs):
    """Build a registered framework around a registered backbone."""
    if name not in FRAMEWORK_REGISTRY:
        raise ValueError(f"Invalid framework {name}")
    if name == "vae":
        kwargs["variational"] = True
    return FRAMEWORK_REGISTRY[name]( **kwargs)

def get_input_engine(input_engine):
    """Build a registered input engine by config name."""
    if input_engine["name"] not in INPUT_ENGINE_REGISTRY:
        raise ValueError(f"Invalid input engine {input_engine['name']}")
    return INPUT_ENGINE_REGISTRY[input_engine.pop("name")](**input_engine)

def engines(engine):
    """Build a registered input engine by config name."""
    if engine["name"] not in INPUT_ENGINE_REGISTRY:
        raise ValueError(f"Invalid input engine {engine['name']}")
    return INPUT_ENGINE_REGISTRY[engine.pop("name")](**engine)

def encoders(encoder):
    """Build a registered encoder by config name."""
    if encoder["name"] not in ENCODER_REGISTRY:
        raise ValueError(f"Invalid encoder {encoder['name']}")
    return ENCODER_REGISTRY[encoder.pop("name")](**encoder)

def actions(action):
    """Build a registered action module by config name."""
    if action["name"] not in ACTION_REGISTRY:
        raise ValueError(f"Invalid action {action['name']}")
    return ACTION_REGISTRY[action.pop("name")](**action)

def decoders(decoder):
    """Build a registered decoder by config name."""
    if decoder["name"] not in DECODER_REGISTRY:
        raise ValueError(f"Invalid decoder {decoder['name']}")
    return DECODER_REGISTRY[decoder.pop("name")](**decoder)

def score_functions(score_mode):
    """Build a registered score function by config name."""
    if score_mode["name"] not in SCORE_FUNCTIONS:
        raise ValueError(f"Invalid score mode {score_mode}")
    return SCORE_FUNCTIONS[score_mode.pop("name")](**score_mode)