"""Compare three saved window-only VUS measurements with both buffer conventions."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import metrics.vus.metrics as vus_metrics
from metrics.vus.analysis.robustness_eval import generate_curve_t


def main():
    original = vus_metrics.generate_curve_t
    rows = []
    try:
        # Preserve the current wrapper's argument mapping, but pin the underlying
        # integration to CPU/float64 rather than automatically choosing CUDA.
        vus_metrics.generate_curve_t = lambda labels, scores, size: generate_curve_t(
            labels, scores, size, device="cpu")
        for key in (
            "ae/default",
            "lewm_encoder/frequency_aux/phase1",
            "lewm_encoder/frequency_aux/cross_attention/observed_query",
        ):
            candidates = list((ROOT / "results/smd" / key / "machine-1-1.txt").rglob("metrics.json"))
            if len(candidates) != 1:
                raise ValueError(f"expected one report for {key}, found {len(candidates)}")
            path = candidates[0]
            report = json.loads(path.read_text())
            with np.load(path.parent / "scores.npz") as saved_arrays:
                scores = saved_arrays["window_scores"]
                labels = saved_arrays["window_labels"]
            options = report["evaluation_settings"]
            buffer = max(1, int(options["eval_window_size"]) // int(options["stride"]))
            current = vus_metrics.get_range_vus_roc(scores, labels, buffer)
            direct = generate_curve_t(labels, scores, buffer, device="cpu")
            saved = report["evaluation"]["window"]
            row = {
                "experiment": key, "machine": "machine-1-1.txt",
                "window_count": len(scores), "requested_buffer": buffer,
                "saved_vus_pr": saved["vus_pr"], "saved_vus_roc": saved["vus_roc"],
                "current_double_cpu_vus_pr": current["VUS_PR"],
                "current_double_cpu_vus_roc": current["VUS_ROC"],
                "direct_single_cpu_vus_pr": direct[-1],
                "direct_single_cpu_vus_roc": direct[-2],
                "double_pr_minus_saved": current["VUS_PR"] - saved["vus_pr"],
                "double_roc_minus_saved": current["VUS_ROC"] - saved["vus_roc"],
                "single_pr_minus_saved": direct[-1] - saved["vus_pr"],
                "single_roc_minus_saved": direct[-2] - saved["vus_roc"],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        vus_metrics.generate_curve_t = original
    output = Path(__file__).with_name("vus_artifact_verification.json")
    output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
