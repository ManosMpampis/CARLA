"""Final model assembly from configuration; independent of training entries.

Training scripts import these builders. Architectures live under models/ and
instantiate through the backbone/framework registries in models/__init__.py.
"""


def get_lewm_model(p):
    """Build the configured LeWM experiment."""
    from models import get_backbone, get_framework

    enc_kwargs = dict(p.get("model_kwargs", {}))
    built = get_backbone(p.get("backbone", "lewm_resnet"), **enc_kwargs)
    aux_kwargs = dict(p.get("aux_kwargs", {}))
    pred_kwargs = dict(p.get("predictor_kwargs", {}))
    return get_framework(
        "lewm",
        encoder=built["model"],
        aux_channels=aux_kwargs.get("aux_channels", (32, 32, 32)),
        stem_channels=pred_kwargs.get("stem_channels", 64),
        neck_widths=tuple(pred_kwargs.get("neck_widths", (64, 64, 64))),
        n_fft=int(pred_kwargs.get("n_fft", 64)),
        hop_length=int(pred_kwargs.get("hop_length", 16)),
        win_length=int(pred_kwargs.get("win_length", 64)),
        aux_kernels=tuple(aux_kwargs.get("kernels", (7, 5, 3))),
        time_steering=pred_kwargs.get("time_steering", True),
        predictor_domain=pred_kwargs.get("domain", "frequency"),
        with_aux=aux_kwargs.get("with_aux", True),
        norm=enc_kwargs.get("norm", "batch"),
        dropout=enc_kwargs.get("dropout", 0.1),
    )


def get_recon_model(p):
    """LeWM encoder + mirrored recon head (+ optional frozen aux)."""
    from models import get_backbone, get_framework
    from models.recon_head import build_mirrored_head
    from models.lewm import TimeAuxiliary

    enc_kwargs = dict(p.get("model_kwargs", {}))
    built = get_backbone(p.get("backbone", "lewm_resnet"), **enc_kwargs)
    encoder = built["model"]
    recon_kwargs = dict(p.get("recon_kwargs", {}))
    head = build_mirrored_head(
        encoder,
        norm=recon_kwargs.get("norm", enc_kwargs.get("norm", "batch")),
        dropout=float(recon_kwargs.get("dropout", enc_kwargs.get("dropout", 0.1))),
    )
    aux = None
    if bool(recon_kwargs.get("with_aux", False)):
        aux_kwargs = dict(p.get("aux_kwargs", {}))
        aux = TimeAuxiliary(
            int(encoder.output_dims),
            aux_channels=tuple(aux_kwargs.get("aux_channels", (32, 32, 32))),
            kernels=tuple(aux_kwargs.get("kernels", (7, 5, 3))),
            norm=enc_kwargs.get("norm", "batch"),
            dropout=enc_kwargs.get("dropout", 0.1),
        )
    model = get_framework("lewm_reconstruction", encoder=encoder, head=head, aux=aux,
                          score_mode=p.get("score_mode", "l1"))
    # Scorer path (utils/reporting.score_with_model) constructs via this
    # builder, so the flag must live on the model itself.
    model.score_aux_crop = bool(p.get("score_aux_crop", False))
    return model


def get_cross_attention_model(p):
    """Build crop attention with an automatically derived encoder mirror."""
    from models import get_backbone, get_framework

    enc_kwargs = dict(p["model_kwargs"])
    encoder = get_backbone(p.get("backbone", "lewm_resnet"),
                           **enc_kwargs)["model"]
    attention_kwargs = dict(p.get("cross_attention_kwargs", {}))
    attention_kwargs.setdefault("norm", enc_kwargs.get("norm", "batch"))
    attention_kwargs.setdefault("dropout", enc_kwargs.get("dropout", 0.0))
    return get_framework("lewm_cross_attention", encoder,
                         **attention_kwargs,
                         score_mode=p.get("score_mode", "l1"))


def _get_baseline_model(p, arm):
    from models import get_backbone, get_framework

    enc_kwargs = dict(p.get("model_kwargs", {}))
    encoder = get_backbone(p.get("backbone", "lewm_resnet"), **enc_kwargs)["model"]
    return get_framework(arm, encoder, model_kwargs=enc_kwargs,
                         recon_kwargs=dict(p.get("recon_kwargs", {})),
                         score_mode=p.get("score_mode", "l1"))


def get_ae_model(p):
    """Build the from-scratch deterministic reconstruction arm."""
    return _get_baseline_model(p, "ae")


def get_vae_model(p):
    """Build the Gaussian-bottleneck reconstruction arm."""
    return _get_baseline_model(p, "vae")
