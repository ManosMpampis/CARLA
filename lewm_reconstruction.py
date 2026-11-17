"""Two-phase LEWM harness: latent pretraining, then mirrored reconstruction.

After LEWM pretraining through this entry point, the same
LeWMResNetEncoder (optionally strided via ``model_kwargs.enc_strides``)
is reused, and a NEW mirrored head (models/recon_head.py: N
ConvTranspose1d at the mirrored rates + final 1x1 conv to input channels)
is trained to reconstruct the normal input with mean L1 over C x W.

Default inference is encoder + head only (fully convolutional,
timestep-agnostic for any W with W % S == 0). The aux-crop variant keeps
the frozen pretext TimeAuxiliary to propose latent crops at inference
(see ReconModel.score); scoring uses configurable L1/L2/MSE reduction.
"""
import argparse
from lewm import _device, run_train, set_seed

from utils.config import create_config

def main(args, update_dictionary=None):
    # TODO: Implement a way to make the new X phases config
    p = create_config(args.config_env, args.config_exp, args.fname, args.version,
                      update_dictionary=update_dictionary)
    set_seed(int(p.get("seed", 4)))
    
    phases = [phase for phase in p.get("phases")]
    assert len(phases) == 2, "lewm reconstruction framework need two phases, one to train the encoder and one ae"
    assert phases[1]["framework"]["pretrained_encoder"] == True, "Second phase of the lewm reconstructor need to load pretrained_encoder"
    for phase in phases:
        run_train(phase, _device(p))
    return

def cli():
    parser = argparse.ArgumentParser(description="Two-phase LEWM reconstruction")
    parser.add_argument("--config_env", required=True, help="Config file for the environment")
    parser.add_argument("--config_exp", required=True, help="Config file for the experiment")
    parser.add_argument("--fname", help="File name of the dataset machine", default="")
    parser.add_argument("--version", help="Experiment version", type=str)
    parser.add_argument("--score", action="store_true", help="score saved weights using this experiment config")
    parser.add_argument("--score_checkpoint")
    
    parser.add_argument("--pretrained_from")
    parser.add_argument("--phase1-version", dest="phase1_version", help="shared encoder run; defaults to --version")
    main(parser.parse_args())


if __name__ == "__main__":
    cli()
