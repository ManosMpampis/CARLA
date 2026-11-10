"""One calibration/evaluation flow for individual series and pooled models."""

import json
from pathlib import Path

import numpy as np

from data.jepa_dataset import JEPADataset, JEPACorpusDataset
from utils.reconstruction_baselines import evaluate_aggregations
from utils.scoring import model_evaluation_options


def _single(model, config, device, train_data, test_data, write):
    options = model_evaluation_options(config, model)
    evaluation, scores, non_overlap, non_overlap_stride = evaluate_aggregations(
        model, train_data.full_train_series, test_data.series, test_data.targets,
        config, device, options=options)
    coverage = {"scored_timesteps": int((scores["cover_counts"] > 0).sum()),
                "total_timesteps": len(test_data.series),
                "first_scored_timestep": int(scores["starts"].min()),
                "input_window": options["wsz"],
                "output_window": int(scores["ends"][0] - scores["starts"][0]),
                "series_start": int(getattr(test_data, "series_start", 0))}
    if non_overlap is not None:
        coverage["non_overlap_window"] = {
            "stride": non_overlap_stride,
            "scored_timesteps": int((non_overlap["cover_counts"] > 0).sum()),
            "window_count": len(non_overlap["window_scores"])}
    report = {"framework": config["framework"],
              "experiment_name": config.get("experiment_name", "default"),
              "run": config.get("version"), "dataset": config["train_db_name"],
              "collection": config.get("dataset_collection", config["train_db_name"]),
              "fname": config["fname"], "evaluation": evaluation,
              "calibration_source": "full clean training series",
              "evaluation_settings": options, "coverage": coverage,
              "feature_schema": list(train_data.feature_schema),
              "preprocessing": train_data.preprocessing}
    calibration = {"source": "full clean training series",
                   "quantile": float(config.get("calibration_quantile", 0.99)),
                   "thresholds": {mode: value["calibrated"]["threshold"]
                                  for mode, value in evaluation.items()}}
    if write:
        directory = Path(config["jepa_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        from utils.evaluation_artifacts import write_final_artifacts
        report["figures"] = write_final_artifacts(
            directory, evaluation, scores, test_data.targets, non_overlap)
        np.savez_compressed(directory / "scores.npz", labels=test_data.targets,
                            **{k: v for k, v in scores.items() if isinstance(v, np.ndarray)})
        (directory / "metrics.json").write_text(json.dumps(report, indent=2))
        (directory / "calibration.json").write_text(json.dumps(calibration, indent=2))
    return report


def _macro(values):
    """Average metrics across machines without concatenating event timelines."""
    result = {}
    for key in values[0]:
        if key in ("threshold", "calibration_threshold", "calibration_quantile"):
            continue  # thresholds belong to the individual calibrated machines
        entries = [value[key] for value in values if key in value]
        if entries and all(isinstance(v, dict) for v in entries):
            result[key] = _macro(entries)
        elif entries and all(v is None or (isinstance(v, (float, int)) and not isinstance(v, bool)) for v in entries):
            finite = [float(v) for v in entries if v is not None and np.isfinite(v)]
            result[key] = float(np.mean(finite)) if finite else None
            result[f"{key}_valid_members"] = len(finite)
    return result


def evaluate_run(model, config, device, *, datasets=None, write=False):
    if datasets is None:
        from utils.common_config import get_jepa_datasets
        train_data, _ = get_jepa_datasets(config)
        test_data = None if isinstance(train_data, JEPACorpusDataset) else JEPADataset(config, train=False)
    else:
        train_data, test_data = datasets
    model.eval()
    if not isinstance(train_data, JEPACorpusDataset):
        return _single(model, config, device, train_data, test_data, write)
    reports = {}
    for name in train_data.machine_files:
        member = {**config, "fname": name,
                  "jepa_dir": str(Path(config["jepa_dir"]) / "machines" / name)}
        member_train = JEPADataset(member, train=True)
        member_test = JEPADataset(member, train=False)
        reports[name] = _single(model, member, device, member_train, member_test, write)
    report = {"framework": config["framework"], "fname": "all-one", "dataset": "smd",
              "aggregation": "macro_per_machine", "shared_checkpoint": config.get("score_checkpoint", config.get("jepa_model")),
              "evaluation": _macro([v["evaluation"] for v in reports.values()]),
              "machines": reports, "machine_count": len(reports)}
    if write:
        (Path(config["jepa_dir"]) / "metrics.json").write_text(json.dumps(report, indent=2))
        (Path(config["jepa_dir"]) / "calibration.json").write_text(json.dumps({
            "source": "per-machine clean training series",
            "machines": {name: {mode: values["calibrated"]["threshold"]
                                for mode, values in item["evaluation"].items()}
                         for name, item in reports.items()}}, indent=2))
    return report
