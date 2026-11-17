"""Final model assembly from configuration; independent of training entries.

Training scripts import these builders. Architectures live under models/ and
instantiate through the backbone/framework registries in models/__init__.py.
"""
from models import get_encoder, get_framework
from models.recon_head import build_mirrored_head
from models.lw_reconstructor import TimeAuxiliary

def get_model(framework):
    name = framework.pop("name")
    encoder_kwargs = framework.pop("encoder", {"name": "resnet"})
    encoder = get_encoder(encoder_kwargs.pop("name", "resnet"), encoder_kwargs)
    return get_framework(name, encoder=encoder, **framework)

def get_recon_model(p):
    """LeWM encoder + mirrored recon head (+ optional frozen aux)."""
    enc_kwargs = dict(p.get("model_kwargs", {}))
    built = get_encoder(p.get("backbone", "resnet"), **enc_kwargs)
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
    
    model.score_aux_crop = bool(p.get("score_aux_crop", False))
    return model

def _get_baseline_model(p, arm):
    enc_kwargs = dict(p.get("model_kwargs", {}))
    encoder = get_encoder(p.get("backbone", "resnet"), **enc_kwargs)["model"]
    return get_framework(arm, encoder, model_kwargs=enc_kwargs,
                         recon_kwargs=dict(p.get("recon_kwargs", {})),
                         score_mode=p.get("score_mode", "l1"))


def get_ae_model(p):
    """Build the from-scratch deterministic reconstruction arm."""
    return _get_baseline_model(p, "ae")


def get_vae_model(p):
    """Build the Gaussian-bottleneck reconstruction arm."""
    return _get_baseline_model(p, "vae")
