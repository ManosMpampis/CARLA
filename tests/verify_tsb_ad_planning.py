"""Reject incompatible benchmark crops during planning, preserving user settings."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.tsb_ad import TSBSeries
from utils.tsb_ad_benchmark import build_benchmark_plan


def main():
    with tempfile.TemporaryDirectory(prefix="tsb-planning-") as temporary:
        root = Path(temporary)
        manifest = root / "experiments.yml"
        manifest.write_text(yaml.safe_dump({
            "common": {"wsz": 512, "stride": 10, "val_fraction": 0.1,
                       "batch_size": 64, "model_kwargs": {"enc_strides": [2, 2, 2]}},
            "experiments": [
                {"framework": "lewm_encoder", "experiment_name": "frequency_aux", "runner": "lewm",
                 "config": str(REPO / "configs/lewm_encoder/frequency_aux/phase1.yml")},
                {"framework": "cross_attention", "experiment_name": "explicit", "runner": "cross_attention",
                 "config": str(REPO / "configs/lewm_encoder/frequency_aux/cross_attention/observed_query.yml"),
                 "pretrained_experiment": "lewm_encoder/frequency_aux",
                 "overrides": {"cross_attention_kwargs": {"context_crop": [0, 513],
                              "target_crop": [513, 769], "position_encoding": "sinusoidal"}}},
            ],
        }))

        def plan(train_end, **kwargs):
            series = TSBSeries(root / f"toy_tr_{train_end}_1st_5000.csv",
                               np.zeros((6000, 2)), np.zeros(6000, dtype=int),
                               train_end, ("a", "b"))
            return build_benchmark_plan(manifest, REPO / "configs/env.yml", "check",
                                        series, root / "results", **kwargs)

        encoder, head = plan(4055)
        assert "benchmark_planning_error" not in encoder.config
        assert "benchmark_planning_error" in head.config, (
            "dry-run accepted explicit crops outside the resolved input window")
        assert "crop" in head.config["benchmark_planning_error"].lower()
        assert head.config["cross_attention_kwargs"]["context_crop"] == [0, 513]
        assert head.config["cross_attention_kwargs"]["target_crop"] == [513, 769]
        assert head.config["val_fraction"] == 0.1
        assert head.config["cross_attention_kwargs"]["position_encoding"] == "sinusoidal"

        encoder, head = plan(500)
        assert "benchmark_planning_error" not in encoder.config
        assert encoder.config["wsz"] == 496
        assert "benchmark_planning_error" in head.config

        encoder, head = plan(4055, overrides={"cross_attention_kwargs": {
            "context_crop": [0, 128], "target_crop": [128, 256]}})
        assert "benchmark_planning_error" not in head.config
        assert head.config["wsz"] == 512
        assert not (root / "results").exists(), "planning wrote training artifacts"
    print("TSB-AD planning: invalid crops and short prefixes reported without changing geometry")


if __name__ == "__main__":
    main()
