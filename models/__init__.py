"""LeWM module registries and experiment assembly."""

from models.actions import DiscreteAction, EncoderAction, MaskedRepresentationAction, NoAction, TimeAuxiliary
from models.ae_baseline import ReconstructionBaseline
from models.encoders import Encoder, ResNetEncoder, TCNEncoder
from models.input_engines import ForecastEngine, MaskedEngine
from models.lewm import LeWM
from models.lewm_cross_attention import CrossAttentionLeWM, GuidedCrossAttentionPredictor
from models.lewm_cross_attention_tcn import CrossAttentionTCNLeWM
from models.lewm_rec_ca import RecCALeWM
from models.recon_head import MirroredReconHead, build_mirrored_head
from models.recon_model import ReconModel
from models.steped_predictor import FreqPredictor, TimePredictor
from utils.reconstruction_scores import reconstruction_score_map


ENCODER_REGISTRY = {
    "resnet": ResNetEncoder,
    "lewm_resnet": ResNetEncoder,
    "convolutional": Encoder,
    "tcn": TCNEncoder,
}
BACKBONE_REGISTRY = ENCODER_REGISTRY.copy()
INPUT_ENGINE_REGISTRY = {"mask": MaskedEngine, "forecast": ForecastEngine}
ACTION_REGISTRY = {
    "masked": MaskedRepresentationAction,
    "tcn": EncoderAction,
    "convolutional": EncoderAction,
    "discrete": DiscreteAction,
    "none": NoAction,
}
PREDICTOR_REGISTRY = {
    "frequency": FreqPredictor,
    "time": TimePredictor,
    "cross_attention": GuidedCrossAttentionPredictor,
}
HEAD_REGISTRY = {"mirrored": MirroredReconHead}
SCORE_FUNCTIONS = {"residual": reconstruction_score_map}
FRAMEWORK_REGISTRY = {
    "lewm": LeWM,
    "lewm_reconstruction": ReconModel,
    "lewm_cross_attention": CrossAttentionLeWM,
    "lewm_cross_attention_tcn": CrossAttentionTCNLeWM,
    "lewm_rec_ca": RecCALeWM,
    "ae": ReconstructionBaseline,
    "vae": ReconstructionBaseline,
}


def _resolve(registry, name, kind):
    if name not in registry:
        raise ValueError(f"Invalid {kind} {name!r}")
    return registry[name]


def get_framework(name, *args, **kwargs):
    if name in ("ae", "vae"):
        kwargs["variational"] = name == "vae"
    return _resolve(FRAMEWORK_REGISTRY, name, "framework")(*args, **kwargs)


def get_backbone(name, **kwargs):
    encoder = _resolve(BACKBONE_REGISTRY, name, "backbone")(**kwargs)
    return {"model": encoder, "dim": [encoder.output_dims]}


def _legacy_encoder(p):
    return get_backbone(p.get("backbone", "lewm_resnet"), **p.get("model_kwargs", {}))["model"]


def _legacy_lewm_config(p):
    """Translate the phase-one YAML into the same module config as the presets."""
    encoder = dict(p.get("model_kwargs", {}))
    predictor = dict(p.get("predictor_kwargs", {}))
    domain = predictor.pop("domain", "frequency")
    if domain not in ("time", "frequency"):
        raise ValueError(f"Invalid legacy predictor domain {domain!r}")
    if domain == "time":
        for option in ("n_fft", "hop_length", "win_length"):
            predictor.pop(option, None)
    predictor.setdefault("dropout", encoder.get("dropout", 0.1))

    aux = dict(p.get("aux_kwargs", {}))
    with_aux = aux.pop("with_aux", True)
    aux.setdefault("norm", encoder.get("norm", "batch"))
    aux.setdefault("dropout", encoder.get("dropout", 0.1))

    masking = dict(p.get("stage_a", {}).get("masking", {}))
    mode = masking.pop("mode", "none").lower()
    if mode not in ("none", "zero", "subanomaly", "sub_anomaly", "injected"):
        raise ValueError(f"Invalid masking mode {mode!r}")
    return {
        "encoder": {"name": p.get("backbone", "lewm_resnet"), **encoder},
        "engine": {
            "name": "mask",
            "mask_type": "zero" if mode in ("none", "zero") else "injection",
            "sample": mode != "none" and p.get("stage", "pretrain") != "adapt",
            **masking,
        },
        "action": {"name": "masked", "aux_input": "latent", "mask_reduction": "nearest",
                   "aux": aux if with_aux else None},
        "predictor": {"name": domain, **predictor},
    }


def _module_options(config, role, default, freeze):
    options = dict(config.get(role) or {})
    name = options.pop("name", default)
    frozen = options.pop("freeze", role == "backbone" and name != "none")
    if not isinstance(frozen, bool):
        raise ValueError(f"{role}.freeze must be a boolean")
    if frozen and name != "none":
        freeze.append(role)
    return name, options


def _build_action(name, action_cls, options, latent_dim, raw_dim, key_dim):
    if action_cls is MaskedRepresentationAction:
        aux_options = options.pop("aux", {})
        source = options.setdefault("aux_input", "raw")
        aux_dim = raw_dim if source == "raw" else latent_dim
        aux = TimeAuxiliary(aux_dim, **aux_options) if aux_options is not None else None
        return action_cls(latent_dim, aux=aux, **options)
    if action_cls is EncoderAction:
        options.setdefault("hidden_dims", [latent_dim] * 3)
        encoder = ENCODER_REGISTRY[name](latent_dim, out_dims=key_dim, **options)
        return action_cls(encoder)
    return action_cls(**options)


def get_lewm_model(p):
    """Assemble a LeWM from explicit modules or a legacy phase-one config."""
    legacy = "lewm_modules" not in p
    config = _legacy_lewm_config(p) if legacy else p["lewm_modules"]
    freeze = []

    backbone_name, backbone_options = _module_options(config, "backbone", "none", freeze)
    backbone_options.pop("pretrained_from", None)
    backbone_options.pop("checkpoint_prefix", None)
    backbone = (None if backbone_name == "none"
                else _resolve(BACKBONE_REGISTRY, backbone_name, "backbone")(**backbone_options))

    encoder_name, encoder_options = _module_options(config, "encoder", "resnet", freeze)
    encoder_cls = _resolve(ENCODER_REGISTRY, encoder_name, "encoder")
    input_key = "in_channels" if encoder_cls is ResNetEncoder else "in_dim"
    if backbone is not None:
        encoder_options.setdefault(input_key, backbone.output_dims)
    elif input_key not in encoder_options:
        encoder_options[input_key] = p.get("model_kwargs", {}).get("in_channels", 38)
    encoder = encoder_cls(**encoder_options)
    if backbone is not None and encoder.input_dims != backbone.output_dims:
        raise ValueError("encoder input width must match backbone output width")
    latent_dim = encoder.output_dims
    raw_dim = backbone.input_dims if backbone is not None else encoder.input_dims
    stride = encoder.total_stride * getattr(backbone, "total_stride", 1)

    engine_name, engine_options = _module_options(config, "engine", "mask", freeze)
    if engine_name == "forecast":
        engine_options["encoder_strides"] = stride
        engine_options.setdefault("input_window", p.get("wsz"))
    engine = _resolve(INPUT_ENGINE_REGISTRY, engine_name, "engine")(**engine_options)

    predictor_name, predictor_options = _module_options(config, "predictor", "frequency", freeze)
    guided = predictor_name == "cross_attention"
    action_name, action_options = _module_options(config, "action", "tcn" if guided else "masked", freeze)
    action_cls = _resolve(ACTION_REGISTRY, action_name, "action")
    if guided != (action_cls is EncoderAction):
        raise ValueError("the cross_attention predictor requires a TCN or convolutional key action")
    if guided and not isinstance(engine, ForecastEngine):
        raise ValueError("guided cross_attention requires the forecast engine")

    action = _build_action(action_name, action_cls, action_options, latent_dim, raw_dim,
                           predictor_options.get("qk_channels", 8))
    if not guided:
        steering = predictor_options.setdefault("time_steering", bool(action.film_dim))
        if steering and not action.film_dim:
            raise ValueError("time_steering requires action features")
        if isinstance(action, DiscreteAction) and not steering:
            raise ValueError("the discrete action module requires time_steering")
        predictor_options["aux_dim"] = action.film_dim
    predictor = _resolve(PREDICTOR_REGISTRY, predictor_name, "predictor")(
        latent_dim, **predictor_options)

    head_name, head_options = _module_options(config, "head", "none", freeze)
    head = None
    if head_name != "none":
        ladder, strides = encoder.channel_ladder, encoder.enc_strides
        if backbone is not None:
            ladder = [*backbone.channel_ladder, *ladder[1:]]
            strides = [*backbone.enc_strides, *strides]
        head = _resolve(HEAD_REGISTRY, head_name, "head")(ladder, strides, **head_options)
    return LeWM(engine=engine, backbone=backbone, encoder=encoder, action=action,
                predictor=predictor, head=head, freeze=freeze,
                score_domain=config.get("score_domain", "latent"),
                score_mode=p.get("score_mode", "l1"), legacy_optimizer_order=legacy,
                score_fn=_resolve(SCORE_FUNCTIONS, config.get("score_fn", "residual"), "score function"))


def get_recon_model(p):
    """Build the historical dense phase-two reconstruction arm."""
    encoder = _legacy_encoder(p)
    enc_options = p.get("model_kwargs", {})
    recon_options = p.get("recon_kwargs", {})
    norm = enc_options.get("norm", "batch")
    dropout = enc_options.get("dropout", 0.1)
    head = build_mirrored_head(encoder, norm=recon_options.get("norm", norm),
                               dropout=recon_options.get("dropout", dropout))
    aux = None
    if recon_options.get("with_aux", False):
        aux_options = dict(p.get("aux_kwargs", {}))
        aux_options.pop("with_aux", None)
        aux_options.setdefault("norm", norm)
        aux_options.setdefault("dropout", dropout)
        aux = TimeAuxiliary(encoder.output_dims, **aux_options)
    model = ReconModel(encoder, head, aux=aux, score_mode=p.get("score_mode", "l1"))
    model.score_aux_crop = bool(p.get("score_aux_crop", False))
    return model


def get_cross_attention_model(p):
    """Build a historical observed/context query or combined Rec+CA arm."""
    options = dict(p.get("cross_attention_kwargs", {}))
    encoder_options = p.get("model_kwargs", {})
    options.setdefault("norm", encoder_options.get("norm", "batch"))
    options.setdefault("dropout", encoder_options.get("dropout", 0.0))
    framework = p.get("cross_attention_model", "lewm_cross_attention")
    if framework not in ("lewm_cross_attention", "lewm_cross_attention_tcn", "lewm_rec_ca"):
        raise ValueError(f"Invalid cross_attention_model {framework!r}")
    return get_framework(framework, _legacy_encoder(p), **options,
                         score_mode=p.get("score_mode", "l1"))


def _get_baseline_model(p, arm):
    return get_framework(arm, encoder=_legacy_encoder(p),
                         model_kwargs=dict(p.get("model_kwargs", {})),
                         recon_kwargs=dict(p.get("recon_kwargs", {})),
                         score_mode=p.get("score_mode", "l1"))


def get_ae_model(p):
    return _get_baseline_model(p, "ae")


def get_vae_model(p):
    return _get_baseline_model(p, "vae")



def get_model(config, *, input_channels):
    """Construct the configured framework using the dataset's input width."""
    framework = config.get("framework")
    factory = _resolve({"ae": get_ae_model, "vae": get_vae_model, "lewm": get_lewm_model},
                       framework, "training framework")
    p = dict(config)
    if framework == "lewm" and "lewm_modules" in p:
        modules = dict(p["lewm_modules"])
        backbone = modules.get("backbone") or {}
        role = "backbone" if backbone.get("name", "none") != "none" else "encoder"
        options = dict(modules.get(role) or {})
        name = options.get("name", "resnet")
        encoder = _resolve(ENCODER_REGISTRY, name, role)
        width_key = "in_channels" if encoder is ResNetEncoder else "in_dim"
        if options.setdefault(width_key, input_channels) != input_channels:
            raise ValueError(f"{role}.{width_key} must match dataset channels ({input_channels})")
        modules[role] = options
        p["lewm_modules"] = modules
    else:
        options = dict(p.get("model_kwargs", {}))
        encoder = _resolve(BACKBONE_REGISTRY, p.get("backbone", "lewm_resnet"), "encoder")
        width_key = "in_channels" if encoder is ResNetEncoder else "in_dim"
        if options.setdefault(width_key, input_channels) != input_channels:
            raise ValueError(f"encoder.{width_key} must match dataset channels ({input_channels})")
        p["model_kwargs"] = options
    model = factory(p)
    if p["wsz"] % model.total_stride:
        raise ValueError("data.window must be divisible by the model's total stride")
    return model
