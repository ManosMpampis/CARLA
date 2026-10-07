"""Assertions for evaluation-only window placement and aggregation.

Usage: ./venv/bin/python tests/verify_evaluation_mapping.py
"""
import os
import sys
import tempfile
import base64
import json
import re

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from utils.scoring import RunningScorer, Scorer, evaluation_options  # noqa: E402
from utils.decision_trace import write_final_decision_trace  # noqa: E402
from utils.reconstruction_baselines import score_both  # noqa: E402


class PositionModel(torch.nn.Module):
    def __init__(self, output_length=None):
        super().__init__()
        self.output_length = output_length

    def score(self, x):
        length = self.output_length or x.shape[-1]
        row = torch.arange(length, dtype=torch.float32, device=x.device)
        fused = row.expand(x.shape[0], -1)
        return {"fused": fused, "levels": {"L0": fused}, "signals": {}}


class FlipModel(torch.nn.Module):
    def score(self, x):
        starts = x[:, 0, 0]
        values = torch.where(starts == 1, 0.0, 9.0)
        fused = values[:, None].expand(-1, x.shape[-1])
        return {"fused": fused, "levels": {"L0": fused}, "signals": {}}


def main():
    series = np.zeros((10, 1), dtype=np.float32)
    scorer = Scorer(PositionModel(), torch.device("cpu"), batch_size=2)

    # Grid stride really permits nonoverlapping input windows, including when
    # the series has a remainder. No off-grid final window is added.
    disjoint = scorer.score_series(series, 4, 4)
    np.testing.assert_array_equal(disjoint["input_start_idxs"], [0, 4])
    np.testing.assert_array_equal(disjoint["cover_counts"],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    all_windows = scorer.score_series(series, 4, 1)
    np.testing.assert_array_equal(all_windows["input_start_idxs"],
                                  np.arange(7))
    np.testing.assert_array_equal(all_windows["cover_counts"],
                                  [1, 2, 3, 4, 4, 4, 4, 3, 2, 1])

    # The output of a 100-step input can score its second half or future steps.
    long_series = np.zeros((200, 1), dtype=np.float32)
    half = Scorer(PositionModel(50), torch.device("cpu"))
    second_half = half.score_series(long_series, 100, 50,
                                    check_mask=[50, 100])
    np.testing.assert_array_equal(second_half["start_idxs"], [50, 100, 150])
    assert second_half["cover_counts"][49] == 0
    assert second_half["cover_counts"][50] == 1
    future = half.score_series(long_series, 100, 50,
                               check_mask=[100, 150])
    np.testing.assert_array_equal(future["start_idxs"], [100, 150])
    np.testing.assert_array_equal(future["end_idxs"], [150, 200])
    assert not future["cover_counts"][:100].any()
    shifted = half.score_series(long_series, 100, 50,
                                check_mask=[100, 150], latency_offset=-10)
    np.testing.assert_array_equal(shifted["start_idxs"], [90, 140, 190])
    np.testing.assert_array_equal(shifted["end_idxs"], [140, 190, 200])

    # Real-time running mean changes as later full windows arrive, and its
    # final state equals full-series aggregation without storing a matrix.
    running = RunningScorer(PositionModel(), torch.device("cpu"), 4)
    first = running.update(series[:4], 0)
    assert first["scores"][-1] == 3.0
    assert running.score_at(3) == 3.0
    running.update(series[1:5], 1)
    assert running.score_at(3) == 2.5
    for start in range(2, 7):
        running.update(series[start:start + 4], start)
    for timestep in range(10):
        assert running.counts[timestep] == all_windows["cover_counts"][timestep]
        np.testing.assert_allclose(running.score_at(timestep),
                                   all_windows["scores"][timestep])
    try:
        running.update(series[:3], 7)
    except ValueError as exc:
        assert "complete" in str(exc)
    else:
        raise AssertionError("incomplete online input was accepted")

    # A single timestep can flip anomaly -> normal -> anomaly as its mean
    # receives additional contributions from later windows.
    flip_series = np.arange(7, dtype=np.float32)[:, None]
    flipping = RunningScorer(FlipModel(), torch.device("cpu"), 4,
                             threshold=5.0, record_history=True)
    for start in range(4):
        flipping.update(flip_series[start:start + 4], start)
    changes = [event["decision"] for event in flipping.decision_changes
               if event["timestep"] == 3]
    assert changes == [True, False, True]
    assert len(flipping.decision_trace()["updates"]) == 4
    assert flipping.decision_at(3) is True
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "decision_trace.html")
        flipping.save_decision_trace_html(path)
        with open(path) as stream:
            html = stream.read()
        assert "decision-step" in html
        encoded = re.search(r"const encodedTrace = '([^']+)';", html).group(1)
        assert json.loads(base64.b64decode(encoded))["threshold"] == 5.0
    final = Scorer(FlipModel(), torch.device("cpu")).score_series(
        flip_series, 4, 1)
    np.testing.assert_allclose(flipping.score_at(3), final["scores"][3])
    transformed = RunningScorer(
        PositionModel(), torch.device("cpu"), 4, threshold=4.0,
        score_transform=lambda channels: channels["fused"] + channels["L0"],
        series_length=len(series))
    for start in range(7):
        transformed.update(series[start:start + 4], start)
    for timestep in range(len(series)):
        expected = 2 * all_windows["scores"][timestep]
        np.testing.assert_allclose(transformed.score_at(timestep), expected)
        assert transformed.decision_at(timestep) == (expected >= 4.0)
    with tempfile.TemporaryDirectory() as folder:
        config = {"wsz": 4, "stride": 1, "jepa_dir": folder,
                  "decision_trace": {"max_windows": 2}}
        artifact = write_final_decision_trace(
            FlipModel(), torch.device("cpu"), flip_series, config, 5.0,
            threshold_operator="gt")
        assert artifact["recorded_windows"] == 2
        assert artifact["total_windows"] == 4 and artifact["truncated"]
        assert os.path.isfile(artifact["html"])
        with open(artifact["json"]) as stream:
            saved = json.load(stream)
        assert len(saved["updates"]) == 2
        assert saved["updates"][1]["decisions"][2] is False

    # AE/VAE uses exactly the same mapped aggregation and window intervals.
    baseline = score_both(PositionModel(), series, 4, 1, 2, "cpu")
    np.testing.assert_array_equal(baseline["timeseries_scores"],
                                  all_windows["scores"])
    np.testing.assert_array_equal(baseline["starts"], all_windows["start_idxs"])
    np.testing.assert_array_equal(baseline["cover_counts"],
                                  all_windows["cover_counts"])

    options = evaluation_options({"wsz": 100, "stride": 10, "evaluation": {
        "input_window": 50, "stride": 50, "check_mask": [25, 50]}})
    assert options["wsz"] == 50 and options["stride"] == 50
    assert options["include_last_window"] is False
    configured = RunningScorer.from_config(
        PositionModel(25), torch.device("cpu"),
        {"wsz": 100, "stride": 10, "evaluation": {
            "input_window": 50, "stride": 50, "check_mask": [25, 50]}})
    assert configured.input_window == 50 and configured.check_mask == [25, 50]

    try:
        half.score_series(long_series, 100, 50, check_mask=[0, 100])
    except ValueError as exc:
        assert "model output length" in str(exc)
    else:
        raise AssertionError("mismatched check_mask was accepted")

    print("Evaluation mapping OK: stride, output placement, latency, "
          "running mean, and AE/VAE parity.")


if __name__ == "__main__":
    main()
