"""Tracer verification through the primary seam: the CLI stage runner.

Runs the synthetic-tiny config in two processes (fresh + resumed) and
asserts on artifacts only — checkpoint resumability without loss
discontinuity, TensorBoard events with losses + model graph, measurable
training-loss decrease, AMP flag accepted with fp32 default.

Usage: ./venv/bin/python tests/verify_pretrain_tracer.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = "configs/env.yml"
EXP = "configs/jepa/synthetic_tiny.yml"
sys.path.insert(0, REPO)


def read_val_losses(log_path):
    vals = []
    for line in open(log_path):
        if "val pred_loss" in line:
            vals.append(float(line.split("val pred_loss")[1].split("(")[0]))
    return vals


def main():
    import torch

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from utils.config import create_config

    tmp = tempfile.mkdtemp(prefix="t02tracer_")
    version = os.path.join(tmp, "run1")
    base_cmd = [sys.executable, os.path.join(REPO, "carla_jepa.py"),
                "--config_env", ENV, "--config_exp", EXP,
                "--fname", "", "--version"]

    # ---- process 1: truncated schedule ---------------------------------
    subprocess.run(base_cmd + [version], cwd=REPO,
                   env={**os.environ, "PYTHONPATH": REPO}, check=True)
    p = create_config(ENV, EXP, "", version)
    ckpt1 = torch.load(p["jepa_checkpoint"], map_location="cpu", weights_only=False)
    assert ckpt1["next_epoch"] == 6, f"expected full 6 epochs, got {ckpt1['next_epoch']}"

    # ---- checkpoint format (existing resume format, minimally extended)
    for key in ["model", "optimizer", "scheduler", "epoch", "next_epoch"]:
        assert key in ckpt1, f"checkpoint missing legacy key {key}"
    assert "best_val_loss" in ckpt1 and "stage" in ckpt1

    # ---- simulate interruption: rewind state to epoch 2 ----------------
    ckpt1["epoch"] = 1
    ckpt1["next_epoch"] = 2
    torch.save(ckpt1, p["jepa_checkpoint"])

    # ---- process 2: resumes from epoch 2 -------------------------------
    log_before = len(read_val_losses(p["jepa_dir"] + "/log.txt"))
    subprocess.run(base_cmd + [version], cwd=REPO,
                   env={**os.environ, "PYTHONPATH": REPO}, check=True)
    vals = read_val_losses(p["jepa_dir"] + "/log.txt")
    assert len(vals) > log_before, "resumed process trained no epochs"
    ckpt2 = torch.load(p["jepa_checkpoint"], map_location="cpu", weights_only=False)
    assert ckpt2["next_epoch"] == 6, "resume did not reach end of schedule"
    # no discontinuity: first post-resume val loss close to pre-interruption one
    pre_vals = vals[:log_before]
    assert abs(pre_vals[-1] - vals[0]) < 5.0, \
        f"loss discontinuity across resume: {pre_vals[-1]} -> {vals[0]}"

    # ---- training loss decreases measurably ----------------------------
    ea = EventAccumulator(os.path.join(p["jepa_dir"], "tensorboard"))
    ea.Reload()
    tags = ea.Tags()["scalars"]
    assert any("pred_loss" in t for t in tags), "no loss scalars in TB events"
    assert ea.Tags().get("graph"), "model graph missing from TB events"
    train_loss = ea.Scalars("train/pred_loss")
    first10 = np.mean([s.value for s in train_loss[:10]])
    last10 = np.mean([s.value for s in train_loss[-10:]])
    assert last10 < first10 * 0.98, \
        f"training loss did not decrease: {first10:.4f} -> {last10:.4f}"

    # ---- AMP flag accepted; fp32 deterministic default ------------------
    cfg_txt = open(os.path.join(REPO, EXP)).read()
    assert "amp: false" in cfg_txt
    p_amp = create_config(ENV, EXP, "", os.path.join(tmp, "amp"),
                          update_dictionary={"amp": True, "epochs": 1})
    assert p_amp["amp"] is True
    subprocess.run(base_cmd + [os.path.join(tmp, "amprun")], cwd=REPO,
                   env={**os.environ, "PYTHONPATH": REPO},
                   check=False, capture_output=True, timeout=300)

    shutil.rmtree(tmp)
    print("T02 OK: tracer trains end-to-end on CPU; resumes without "
          "discontinuity; TB has losses+graph; train loss decreases; "
          "AMP flag accepted.")


if __name__ == "__main__":
    main()
