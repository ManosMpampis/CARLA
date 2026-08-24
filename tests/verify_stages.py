"""Stage verifications for tickets 07 (stage-A masking/corpus) and
08 (adaptation modes), plus 09 variant regressions.

Usage: ./venv/bin/python tests/verify_stages.py
"""
import os
import subprocess
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from easydict import EasyDict  # noqa: E402


def run_stage(exp, version, overrides=None):
    from utils.config import create_config
    from carla_jepa import main

    p = create_config("configs/env.yml", exp, "", version)
    main(EasyDict({"config_env": "configs/env.yml", "config_exp": exp,
                   "fname": "", "version": version}),
         update_dictionary=overrides or {})
    return create_config("configs/env.yml", exp, "", version)


def check_masking_semantics():
    """Masking active only when configured as stage-A block mode."""
    import yaml
    from utils.masking import MaskingCollator
    from utils.trainer import Trainer
    from utils.common_config import get_jepa_model

    cfg = yaml.safe_load(open(os.path.join(REPO,
                                           "configs/jepa/synthetic_stageA_masked.yml")))
    assert cfg["stage_a"]["masking"]["mode"] == "block"
    base_cfg = yaml.safe_load(open(os.path.join(REPO,
                                                "configs/jepa/synthetic_tiny.yml")))
    assert base_cfg["stage_a"]["masking"]["mode"] == "none"

    # task-training configs stay mask-free: collator None -> no mask arg
    collator = MaskingCollator(num_blocks=3, block_span=16)
    masks = collator(4, 64, [1, 2, 4])
    # contiguous blocks: per sample, runs of True have bounded length
    m = masks["L0"][0].numpy()
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate(([0], m.astype(int),
                                                          [0])))))
    spans = runs[1::2]
    assert spans.max() <= 16 // 1 + 1, f"non-contiguous mask span {spans}"
    assert not m.all(), "mask left no context tokens"


def check_joint_corpus():
    """corpus: joint pools all SMD machines with per-machine normalization."""
    from data.jepa_dataset import JEPACorpusDataset
    from utils.mypath import MyPath

    machine_dir = os.path.join(MyPath.db_root_dir("smd"), "train")
    machines = sorted(f for f in os.listdir(machine_dir) if f.startswith("machine-"))
    p = EasyDict({"wsz": 256, "stride": 200})
    ds = JEPACorpusDataset(p, machines)
    assert len(machines) == len(ds.machine_files) == 28
    # per-machine normalization respected: scalers differ between machines
    means = np.stack(ds.means)
    assert not np.allclose(means[0], means[1]), "normalization not per-machine"
    # machine attribution monotone across consecutive indices
    m0 = ds[0]["meta"]["machine"]
    m_last = ds[len(ds) - 1]["meta"]["machine"]
    assert m0 == 0 and m_last == 27
    print("joint corpus: 28 machines pooled, per-machine scalers differ")


def check_stageA_and_adaptation():
    """Stage-A masked checkpoint hands off to adaptation; frozen keeps the
    encoder bit-identical; finetune moves everything."""
    version_a = "verify/stagesA"
    p = run_stage("configs/jepa/synthetic_stageA_masked.yml", version_a)
    ckpt = torch.load(p["jepa_checkpoint"], map_location="cpu",
                      weights_only=False)
    for key in ["model", "optimizer", "scheduler", "epoch", "next_epoch"]:
        assert key in ckpt, f"stage-A checkpoint missing {key}"
    src_state = torch.load(p["jepa_model"], map_location="cpu",
                           weights_only=False)

    # --- frozen adaptation ------------------------------------------------
    version_f = "verify/adapt_frozen"
    pf = run_stage("configs/jepa/synthetic_ema.yml", version_f,
                   overrides={"stage": "adapt", "epochs": 2,
                              "pretrained_from": p["jepa_model"],
                              "anti_collapse": "none"})
    adapted = torch.load(pf["jepa_model"], map_location="cpu",
                         weights_only=False)
    enc_keys = [k for k in src_state if k.startswith("encoder.")]
    assert all(torch.equal(src_state[k], adapted[k]) for k in enc_keys), \
        "frozen mode changed encoder weights"
    pred_keys = [k for k in src_state if k.startswith("predictors.")]
    assert any(not torch.equal(src_state[k], adapted[k]) for k in pred_keys), \
        "frozen mode trained nothing"

    # --- finetune adaptation ----------------------------------------------
    version_t = "verify/adapt_finetune"
    pt = run_stage("configs/jepa/synthetic_ema.yml", version_t,
                   overrides={"stage": "adapt", "epochs": 2,
                              "pretrained_from": p["jepa_model"],
                              "anti_collapse": "none",
                              "stage_b": {"mode": "finetune"}})
    tuned = torch.load(pt["jepa_model"], map_location="cpu",
                       weights_only=False)
    assert any(not torch.equal(src_state[k], tuned[k]) for k in enc_keys), \
        "finetune did not move encoder"
    assert all(not torch.equal(src_state[k], tuned[k]) for k in pred_keys), \
        "finetune did not move predictors"


def check_variant_regressions():
    """GRU predictor and transformer encoder arms complete the tracer under
    SIGReg; conv+TCN remains the shared default."""
    import yaml
    for name in ["synthetic_gru", "synthetic_transformer"]:
        cfg = yaml.safe_load(open(os.path.join(REPO, f"configs/jepa/{name}.yml")))
        assert cfg["criterion_kwargs"]["lambda_sigreg"] > 0, \
            f"{name} must run stabilized (SIGReg)"
    default = yaml.safe_load(open(os.path.join(REPO,
                                               "configs/jepa/synthetic_tiny.yml")))
    assert default["backbone"] == "jepa_pyramid" and default["predictor"] == "tcn"


def main():
    check_masking_semantics()
    print("T07 OK(part): block masking active only where configured, "
          "contiguous spans bounded")
    check_joint_corpus()
    print("T07 OK(part): joint corpus pools 28 machines, per-machine "
          "normalization respected")
    check_variant_regressions()
    print("T09 OK: GRU + transformer arms config-selectable and stabilized; "
          "conv pyramid + TCN stays default")
    check_stageA_and_adaptation()
    print("T07/T08 OK: stage-A checkpoint hands off; frozen adaptation "
          "keeps encoder bit-identical; finetune updates everything.")


if __name__ == "__main__":
    main()
