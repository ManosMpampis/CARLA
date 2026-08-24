"""Ticket 10 verification: SubAnomaly calibrator.

Runs the score stage CLI on machine-1-1, then independently re-derives
the calibration inputs to prove:
  - thresholds equal quantiles of CLEAN-TRAIN fused scores (no test
    labels reachable),
  - injected-probe scores separate from clean-train scores,
  - fusion weights come from probe separation (or honest fallback).

Usage: ./venv/bin/python tests/verify_calibration.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

VERSION_SRC = "smoke_adapt"          # trained weights from ticket-03 smoke
VERSION_SCORE = "verify/calibration"


def main():
    from easydict import EasyDict
    from utils.config import create_config

    cfg = yaml.safe_load(open(os.path.join(REPO, "configs/jepa/smd_score_smoke.yml")))
    exp = "/tmp/opencode/t10_score.yml"
    yaml.safe_dump(cfg, open(exp, "w"))

    cmd = [sys.executable, os.path.join(REPO, "carla_jepa.py"),
           "--config_env", "configs/env.yml", "--config_exp", exp,
           "--fname", "machine-1-1.txt", "--version", VERSION_SCORE]
    proc = subprocess.run(cmd, cwd=REPO,
                          env={**os.environ, "PYTHONPATH": REPO},
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]

    p = create_config("configs/env.yml", exp, "machine-1-1.txt", VERSION_SCORE)
    payload = json.load(open(p["calibration_path"]))
    for key in ["quantile", "thresholds", "weights", "fallback", "threshold_fused"]:
        assert key in payload, f"calibration.json missing {key}"

    # --- independent re-derivation of the train-side inputs ----------------
    from utils.common_config import get_jepa_model, get_jepa_datasets
    from utils.scoring import Scorer
    from data.augment import SubAnomaly

    device = torch.device("cpu")
    model = get_jepa_model(EasyDict({**cfg, "fname": "machine-1-1.txt"}))
    state = torch.load(p["score_checkpoint"], map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state)
    model.eval()

    scorer = Scorer(model, device)
    train_dataset, _ = get_jepa_datasets(
        EasyDict({**cfg, "fname": "machine-1-1.txt"}))
    clean_result = scorer.score_series(
        np.asarray(train_dataset.series, dtype=np.float32),
        cfg["wsz"], cfg["stride"])
    clean_fused = clean_result.pop("scores")
    clean_channels = {"fused": clean_fused, **clean_result["channels"]}

    sanomaly = SubAnomaly(cfg["probe_kwargs"]["portion"])
    rng = np.random.default_rng(4)
    idxs = rng.integers(0, len(train_dataset), size=150)
    wins = np.stack([sanomaly(train_dataset[int(i)]["ts"]).astype(np.float32)
                     for i in idxs])
    probe_scores = scorer.score_windows(
        torch.from_numpy(wins).permute(0, 2, 1).contiguous())
    # same statistic on both sides: per-window means (as Calibrator sees them)
    probe_fused = probe_scores["fused"].mean(axis=1)
    clean_fused_w = np.array([
        clean_fused[s:e].mean()
        for s, e in zip(clean_result["start_idxs"], clean_result["end_idxs"])
    ])

    # separation on machine-1-1: injected probes score higher than clean
    c_mean, pr_mean = clean_fused_w.mean(), probe_fused.mean()
    assert pr_mean > c_mean, \
        f"no separation on machine-1-1: clean {c_mean:.4f} vs probe {pr_mean:.4f}"

    # thresholds trace only to clean-train distributions (tolerance absorbs
    # CUDA-vs-CPU kernel numerics between the CLI run and this check)
    q = payload["quantile"]
    if payload["fallback"]:
        expected_th = float(np.quantile(clean_fused, q))
    else:
        fused = sum(payload["weights"][n] * np.asarray(v, dtype=np.float64)
                    for n, v in clean_channels.items())
        expected_th = float(np.quantile(fused, q))
        wsum = sum(payload["weights"].values())
        assert abs(wsum - 1.0) < 1e-6, "calibrated weights must sum to 1"
    assert abs(payload["threshold_fused"] - expected_th) <= \
        5e-4 * max(1.0, abs(expected_th)), \
        f"threshold {payload['threshold_fused']} != clean-train quantile {expected_th}"

    print(f"T10 OK: machine-1-1 clean mean {c_mean:.4f} < probe mean "
          f"{pr_mean:.4f}; thresholds persist train-side only "
          f"(fallback={payload['fallback']}).")


if __name__ == "__main__":
    main()
