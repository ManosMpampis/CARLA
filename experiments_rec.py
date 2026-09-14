"""Phase-1 + Phase-2 sweep over every SMD machine: pretext -> recon -> score.

Per machine, with the grilled Phase-2 contract (S=4, W=128, W % S == 0):
  1) pretext: carla_steered.py  (steered ResNet encoder + aux + freq predictor)
  2) recon:   carla_recon.py    (same encoder init + mirrored head, dense L1)
  3) score:   carla_recon.py    (shared engine, per-machine train-only 0.99)

Fixed sweep geometry (overrides whatever the base configs say):
  model_kwargs = {in_channels: 38, enc_channels: [32, 64], enc_strides: [2, 2]}
  wsz = 128  (128 % 4 == 0, exact transpose round-trip, no crop logic)
  pretext epochs = 150 (aux head convergence), recon epochs = 50.
  scheduler = warmup (10 epochs, LinearLR) + cosine_restart with
  lr_eta_min = lr / 10, T_period = epochs - 10 (one cosine cycle).

Weight handoff: if the recon/score base config does NOT specify
pretrained_from / score_checkpoint, the path is derived from the saved
folder of the previous stage:
  results/<train_db>/<version>/<fname>/jepa/model.pth.tar
Runs resume automatically (Trainer.resume), so re-running this script
continues interrupted machines instead of restarting them.

Usage:
  ./venv/bin/python experiments_rec.py                                   # full sweep (CUDA, hours)
  ./venv/bin/python experiments_rec.py --dry-run                         # print the plan, train nothing
  ./venv/bin/python experiments_rec.py --limit 2                         # first 2 machines only
  ./venv/bin/python experiments_rec.py --start-from machine-1-3.txt      # resume the loop here
  ./venv/bin/python experiments_rec.py --pretext-version rec_s2 --recon-version rec_s2_recon
"""
import argparse
from html import parser
import os

import yaml
from easydict import EasyDict

from carla_recon import main as recon_main
from carla_steered import main as steered_main

ENV_YML = "configs/env.yml"
PRETEXT_YML = "configs/jepa/steering/steered_lewm_pretrain.yml"
RECON_TRAIN_YML = "configs/jepa/steering/smd_recon_train.yml"
RECON_SCORE_YML = "configs/jepa/steering/smd_recon_score.yml"

# Sweep geometry (grilled contract: S = 2*2 = 4, W = 128 divisible by S).
IN_CHANNELS = 38
ENC_CHANNELS = [32, 64]
ENC_STRIDES = [2, 2]
WSZ = 128
PRETEXT_EPOCHS = 300
RECON_EPOCHS = 300
WARMUP_EPOCHS = 10
BASE_LR = 0.002


def _load_yml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _root_dir(env_yml=ENV_YML):
    return _load_yml(env_yml)["root_dir"]


def jepa_model_path(version, fname, exp_yml, env_yml=ENV_YML):
    """Saved-folder convention: results/<db>/<version>/<fname>/jepa[tag]/model.pth.tar."""
    cfg = _load_yml(exp_yml)
    tag = cfg.get("tag_jepa")
    jepa_dirname = f"jepa_{tag}" if tag else "jepa"
    return os.path.join(_root_dir(env_yml), cfg.get("train_db_name", "smd"),
                        version, fname, jepa_dirname, "model.pth.tar")


def _cosine_restart_patch(epochs, lr=BASE_LR):
    """Warmup-10 + cosine-restart, minimum = lr / 10 (one full cycle)."""
    return {
        "epochs": int(epochs),
        "scheduler": "cosine_restart",
        "scheduler_kwargs": {
            "lr_warmup_epochs": int(WARMUP_EPOCHS),
            "lr_eta_min": float(lr) / 10.0,
            "T_period": int(epochs) - int(WARMUP_EPOCHS),
            "T_mul": 1,
        },
    }


def _model_patch():
    return {
        "wsz": int(WSZ),
        "model_kwargs": {
            "in_channels": int(IN_CHANNELS),
            "enc_channels": list(ENC_CHANNELS),
            "enc_strides": list(ENC_STRIDES),
            "norm": "batch",
            "dropout": 0.1,
        },
    }


def machine_list():
    train_dir = os.path.join("datasets", "SMD", "train")
    files = sorted(f for f in os.listdir(train_dir) if f.startswith("machine-"))
    return files


def run_machine(fname, pretext_version, recon_version, dry_run=False,
                score_aux_crop=False, threshold_per_channel=False, threshold_channels_and=False):
    """Pretext (150) -> recon (50) -> score for one machine file."""
    # -- 1) pretext ------------------------------------------------------
    pretext_patch = {"stage": "pretrain"}
    pretext_patch.update(_model_patch())
    pretext_patch.update(_cosine_restart_patch(PRETEXT_EPOCHS))
    pretext_args = EasyDict({"config_env": ENV_YML, "config_exp": PRETEXT_YML,
                             "fname": fname, "version": pretext_version})
    pretext_model = jepa_model_path(pretext_version, fname, PRETEXT_YML)

    # -- 2) recon --------------------------------------------------------
    # Fallback: pretrained_from comes from the saved pretext folder unless
    # the recon base config already specifies it.
    recon_cfg = _load_yml(RECON_TRAIN_YML)
    pretrained_from = recon_cfg.get("pretrained_from") or pretext_model
    recon_patch = {"stage": "recon", "pretrained_from": pretrained_from,
                   "recon_kwargs": {"with_aux": True, "norm": "batch",
                                    "dropout": 0.1}}
    recon_patch.update(_model_patch())
    recon_patch.update(_cosine_restart_patch(RECON_EPOCHS))
    recon_args = EasyDict({"config_env": ENV_YML, "config_exp": RECON_TRAIN_YML,
                           "fname": fname, "version": recon_version})
    recon_model = jepa_model_path(recon_version, fname, RECON_TRAIN_YML)

    # -- 3) score --------------------------------------------------------
    # Fallback: score_checkpoint comes from the saved recon folder unless
    # the score base config already specifies it.
    score_cfg = _load_yml(RECON_SCORE_YML)
    score_checkpoint = score_cfg.get("score_checkpoint") or recon_model
    score_patch = {"stage": "score", "score_checkpoint": score_checkpoint,
                   "score_aux_crop": score_aux_crop,
                   "threshold_per_channel": threshold_per_channel,
                   "threshold_channel_operator": "and" if threshold_channels_and else "or",
                   "recon_kwargs": {"with_aux": True, "norm": "batch",
                                    "dropout": 0.1}}
    score_patch.update(_model_patch())
    score_args = EasyDict({"config_env": ENV_YML, "config_exp": RECON_SCORE_YML,
                           "fname": fname, "version": recon_version})

    plan = [(f"pretext [{fname}]", pretext_args, pretext_patch),
            (f"recon   [{fname}]", recon_args, recon_patch),
            (f"score   [{fname}]", score_args, score_patch)]
    if dry_run:
        for name, args, patch in plan:
            print(f"DRY {name} version={args.version} "
                  f"epochs={patch.get('epochs', '-')} "
                  f"sched={patch.get('scheduler', 'score-stage')}")
        print(f"  pretrained_from -> {pretrained_from}")
        print(f"  score_checkpoint -> {score_checkpoint}")
        return

    print(f"=== pretext {fname} (epochs={PRETEXT_EPOCHS}) ===", flush=True)
    steered_main(pretext_args, update_dictionary=dict(pretext_patch))
    print(f"=== recon {fname} (epochs={RECON_EPOCHS}) ===", flush=True)
    recon_main(recon_args, update_dictionary=dict(recon_patch))
    print(f"=== score {fname} (aux_crop={score_aux_crop}) ===", flush=True)
    recon_main(score_args, update_dictionary=dict(score_patch))


def main():
    parser = argparse.ArgumentParser(description="Rec sweep: pretext->recon->score per SMD machine")
    parser.add_argument("--pretext-version", default="rec_s2")
    parser.add_argument("--recon-version", default="rec_s2_recon")
    parser.add_argument("--limit", type=int, default=0, help="first N machines only (0 = all)")
    parser.add_argument("--start-from", default="", help="machine file to resume from (inclusive)")
    parser.add_argument("--threshold-per-channel", action="store_true",
                        help="threshold each reconstructed SMD channel independently")
    parser.add_argument("--score-aux-crop", action="store_true",
                        help="score the aux-crop variant instead of full windows")
    parser.add_argument("--threshold-channels-and", action="store_true",
                        help="require every channel threshold to be exceeded")
    parser.add_argument("--all", action="store_false",
                        help="run experiment fitted in all sub-datasets")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = machine_list()
    print(f"{len(files)} machines: {files[0]} .. {files[-1]}")
    if args.all:
        files=["all"]
    else:
        if args.start_from:
            idx = files.index(args.start_from)
            files = files[idx:]
        if args.limit and args.limit > 0:
            files = files[:args.limit]
    for fname in files:
        run_machine(fname, args.pretext_version, f"{args.recon_version}{"_crop" if args.score_aux_crop else ""}",
                    dry_run=args.dry_run, score_aux_crop=args.score_aux_crop, threshold_per_channel=args.threshold_per_channel,
                    threshold_channels_and=args.threshold_channels_and)


if __name__ == "__main__":
    main()
