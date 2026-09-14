"""Run the steered pretext -> reconstruction -> score experiment on PSM.

PSM has one train/test series rather than SMD's per-machine files.  The
underlying stage entry points and reconstruction setup are otherwise the same
as ``experiments_rec.py``.

Examples::

    ./venv/bin/python experiments_psm.py --dry-run
    ./venv/bin/python experiments_psm.py --version psm_rec
    ./venv/bin/python experiments_psm.py --score-aux-crop
"""
import argparse
import os

import yaml
from easydict import EasyDict

from carla_recon import main as recon_main
from carla_steered import main as steered_main


ENV_YML = "configs/env.yml"
PRETEXT_YML = "configs/jepa/steering/steered_lewm_pretrain.yml"
RECON_TRAIN_YML = "configs/jepa/steering/smd_recon_train.yml"
RECON_SCORE_YML = "configs/jepa/steering/smd_recon_score.yml"

DATASET = "psm"
FNAME = "psm"
IN_CHANNELS = 25
WSZ = 1024
STRIDE = 10
ENC_CHANNELS = [32, 64]
ENC_STRIDES = [2, 2]
PRETEXT_EPOCHS = 300
RECON_EPOCHS = 600
BASE_LR = 0.002
WARMUP_EPOCHS = 10


def _load_yml(path):
    with open(path) as stream:
        return yaml.safe_load(stream)


def _model_patch():
    return {
        "train_db_name": DATASET,
        "val_db_name": DATASET,
        "fname": FNAME,
        "wsz": WSZ,
        "stride": STRIDE,
        "model_kwargs": {
            "in_channels": IN_CHANNELS,
            "enc_channels": list(ENC_CHANNELS),
            "enc_strides": list(ENC_STRIDES),
            "norm": "batch",
            "dropout": 0.1,
        },
    }


def _schedule_patch(epochs):
    return {
        "epochs": epochs,
        "scheduler": "cosine_restart",
        "scheduler_kwargs": {
            "lr_warmup_epochs": WARMUP_EPOCHS,
            "lr_eta_min": BASE_LR / 10.0,
            "T_period": epochs - WARMUP_EPOCHS,
            "T_mul": 1,
        },
    }


def _model_path(version, config_path, env_path=ENV_YML):
    cfg = _load_yml(config_path)
    tag = cfg.get("tag_jepa")
    dirname = f"jepa_{tag}" if tag else "jepa"
    root = _load_yml(env_path)["root_dir"]
    return os.path.join(root, DATASET, version, FNAME, dirname, "model.pth.tar")


def run(pretext_version, recon_version, dry_run=False, score_aux_crop=False,
        threshold_per_channel=False, threshold_channels_and=False):
    pretext_model = _model_path(pretext_version, PRETEXT_YML)
    recon_model = _model_path(recon_version, RECON_TRAIN_YML)

    pretext_patch = {"stage": "pretrain", **_model_patch(),
                     **_schedule_patch(PRETEXT_EPOCHS)}
    recon_patch = {
        "stage": "recon",
        "pretrained_from": pretext_model,
        "recon_kwargs": {"with_aux": True, "norm": "batch", "dropout": 0.1},
        **_model_patch(),
        **_schedule_patch(RECON_EPOCHS),
    }
    score_patch = {
        "stage": "score",
        "score_checkpoint": recon_model,
        "score_aux_crop": score_aux_crop,
        "threshold_per_channel": threshold_per_channel,
        "threshold_channel_operator": "and" if threshold_channels_and else "or",
        "save_timeseries_plot": True,
        "recon_kwargs": {"with_aux": True, "norm": "batch", "dropout": 0.1},
        **_model_patch(),
    }
    pretext_args = EasyDict({"config_env": ENV_YML, "config_exp": PRETEXT_YML,
                             "fname": FNAME, "version": pretext_version})
    recon_args = EasyDict({"config_env": ENV_YML, "config_exp": RECON_TRAIN_YML,
                           "fname": FNAME, "version": recon_version})
    score_args = EasyDict({"config_env": ENV_YML, "config_exp": RECON_SCORE_YML,
                           "fname": FNAME, "version": recon_version})

    if dry_run:
        print(f"DRY pretext: {PRETEXT_EPOCHS} epochs")
        print(f"DRY recon:   {RECON_EPOCHS} epochs")
        print(f"DRY score:   aux_crop={score_aux_crop}, "
              f"per_channel={threshold_per_channel}")
        print(f"pretrained_from -> {pretext_model}")
        print(f"score_checkpoint -> {recon_model}")
        return

    # print(f"=== pretext PSM (epochs={PRETEXT_EPOCHS}) ===", flush=True)
    # steered_main(pretext_args, update_dictionary=pretext_patch)
    # print(f"=== recon PSM (epochs={RECON_EPOCHS}) ===", flush=True)
    # recon_main(recon_args, update_dictionary=recon_patch)
    print(f"=== score PSM (aux_crop={score_aux_crop}) ===", flush=True)
    recon_main(score_args, update_dictionary=score_patch)


def main():
    parser = argparse.ArgumentParser(description="Run reconstruction experiment on PSM")
    parser.add_argument("--pretext-version", default="lewm_rec/psm_pretext")
    parser.add_argument("--recon-version", default="lewm_rec/psm_recon")
    parser.add_argument("--score-aux-crop", action="store_true")
    parser.add_argument("--threshold-per-channel", action="store_true",
                        help="threshold each reconstructed PSM channel independently")
    parser.add_argument("--threshold-channels-and", action="store_true",
                        help="require every channel threshold to be exceeded")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.pretext_version, args.recon_version,
        dry_run=args.dry_run, score_aux_crop=args.score_aux_crop,
        threshold_per_channel=args.threshold_per_channel,
        threshold_channels_and=args.threshold_channels_and)


if __name__ == "__main__":
    main()
