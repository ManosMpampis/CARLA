"""One reporting vocabulary for every experiment and evaluation mode."""

import csv
import json
from pathlib import Path

MODES = {"window": "Test_window", "non_overlap_window": "Test_non_overlap",
         "timeseries": "Test_timeseries"}
DECISION_METRICS = ("event_f1", "f1", "range_f1", "affiliation_f1", "pa_f1")
SCORE_METRICS = ("vus_pr", "vus_roc", "auc_pr", "auc_roc")


def evaluation_row(metrics, **info):
    row = dict(info)
    for source in ("calibrated", "oracle"):
        values = metrics.get(source, {})
        for name in ("threshold", "calibration_quantile", *DECISION_METRICS[:-1]):
            if name in values:
                row[f"{source}_{name}"] = values[name]
    row.update({name: metrics[name] for name in SCORE_METRICS if name in metrics})
    # PA is always last in exports, as well as in metric computation.
    for source in ("calibrated", "oracle"):
        if "pa_f1" in metrics.get(source, {}):
            row[f"{source}_pa_f1"] = metrics[source]["pa_f1"]
    return row


def append_csv(path, row):
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def log_evaluation(writer, evaluation, epoch):
    for protocol, section in MODES.items():
        if protocol not in evaluation:
            continue
        metrics = evaluation[protocol]
        for name in SCORE_METRICS:
            if metrics.get(name) is not None:
                writer.add_scalar(f"{section}/{name}", metrics[name], epoch)
        for source in ("calibrated", "oracle"):
            values = metrics[source]
            if values.get("threshold") is not None:
                writer.add_scalar(f"{section}/{source}/threshold", values["threshold"], epoch)
            if values.get("calibration_quantile") is not None:
                writer.add_scalar(f"{section}/{source}/quantile", values["calibration_quantile"], epoch)
            for name in DECISION_METRICS[:-1]:
                value = metrics[source].get(name)
                if value is not None:
                    writer.add_scalar(f"{section}/{source}/{name}", value, epoch)
        for source in ("calibrated", "oracle"):
            if metrics[source].get("pa_f1") is not None:
                writer.add_scalar(f"{section}/{source}/pa_f1", metrics[source]["pa_f1"], epoch)


def record_evaluation(config, evaluation, epoch, log):
    directory = Path(config["jepa_dir"])
    with (directory / "test_evaluations.jsonl").open("a") as stream:
        stream.write(json.dumps({"epoch": epoch, "evaluation": evaluation}) + "\n")
    for protocol, section in MODES.items():
        if protocol not in evaluation:
            continue
        row = evaluation_row(evaluation[protocol], epoch=epoch)
        append_csv(directory / f"{section}.csv", row)
        fields = ", ".join(f"{name}={value:.6f}" for name, value in row.items()
                           if name != "epoch" and value is not None)
        log(f"{section} epoch {epoch} | {fields}")


def loss_metrics(losses, *, reconstruction_key="pred_loss"):
    """Retain components once, naming the prediction/reconstruction objective."""
    result = {}
    reconstruction = losses.get(reconstruction_key)
    for name, value in losses.items():
        if name == reconstruction_key:
            result["reconstruction loss"] = value
        elif name in {"pred_loss", "mse", "l1"} and value is reconstruction:
            continue  # ReconL1Loss exposes the same tensor under three aliases.
        else:
            result[name] = value
    return result


def latent_variance(latents):
    """Mean per-channel population variance, averaged over latent levels."""
    import torch
    levels = latents.values() if isinstance(latents, dict) else (latents,)
    return torch.stack([z.detach().float().var(dim=(0, 2), unbiased=False).mean()
                        for z in levels]).mean()


def log_losses(writer, split, losses, epoch):
    for name, value in losses.items():
        writer.add_scalar(f"{split}/{name}", value, epoch)


def record_losses(config, split, losses, epoch, log):
    append_csv(Path(config["jepa_dir"]) / f"{split}.csv", {"epoch": epoch, **losses})
    log(f"{split} epoch {epoch} | " + ", ".join(
        f"{name}={value:.6f}" for name, value in losses.items()))


def truncate_epoch_history(directory, completed_epoch):
    """Discard logs beyond the resumed state before replaying interrupted work."""
    directory = Path(directory)
    for name in ("train", "validation", *MODES.values()):
        path = directory / f"{name}.csv"
        if not path.exists():
            continue
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = [row for row in reader if int(row["epoch"]) <= completed_epoch]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    for name in ("epochs.jsonl", "test_evaluations.jsonl"):
        path = directory / name
        if path.exists():
            rows = [line for line in path.read_text().splitlines()
                    if json.loads(line)["epoch"] <= completed_epoch]
            path.write_text("".join(line + "\n" for line in rows))
