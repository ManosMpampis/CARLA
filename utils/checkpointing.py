"""Shared epoch snapshots and independent validation/oracle selections."""

import json
from pathlib import Path

import numpy as np
import torch

from utils.evaluation_logging import MODES


def inference_metadata(config, dataset=None):
    result = {"arm": config.get("arm", config.get("criterion")),
              "model_kwargs": dict(config.get("model_kwargs", {})),
              "recon_kwargs": dict(config.get("recon_kwargs", {})),
              "window_length": config.get("wsz"), "stride": config.get("stride"),
              "score_mode": config.get("score_mode", "l1"),
              "score": "mean_timestep_score_map", "overlap": "cover_count_mean",
              "vae_latent": "posterior_mean" if config.get("arm") == "vae" else None}
    if dataset is not None:
        for name in ("mean", "std"):
            value = getattr(dataset, name, None)
            if value is not None:
                result[f"normalization_{name}"] = np.asarray(value).tolist()
    return result


class Checkpoints:
    def __init__(self, config, model, optimizer, scheduler, *, best=None, extra=None):
        self.config, self.model = config, model
        self.optimizer, self.scheduler = optimizer, scheduler
        self.directory = Path(config["jepa_dir"])
        self.best = {"validation_loss": float("inf"), **(best or {})}
        self.saved_checkpoints = {}
        # A best snapshot can be newer than the latest periodic resume file.
        # Preserve its selection value if an interrupted epoch is replayed.
        best_path = self.directory / "best_metrics.json"
        if best_path.exists():
            saved_best = json.loads(best_path.read_text())
            self.saved_checkpoints = saved_best.get("checkpoints", {})
            for name, value in saved_best.items():
                if name == "validation_loss":
                    self.best[name] = min(self.best[name], value)
                elif name.endswith("_oracle_f1"):
                    self.best[name] = max(self.best.get(name, float("-inf")), value)
        self.extra = extra or {}
        self.save_every = int(config.get("save_every", config.get("checkpoint_every", 10)))
        if self.save_every < 0:
            raise ValueError("save_every must be nonnegative (0 disables periodic snapshots)")

    @staticmethod
    def _save(payload, path):
        """Replace a complete file, leaving the previous checkpoint readable."""
        path = Path(path)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _record_save(self, payload, path):
        self._save(payload, path)
        self.saved_checkpoints[Path(path).name] = {
            "epoch": payload["epoch"], "selection": payload["selection"]}

    def update(self, epoch, losses, evaluation, *, evaluated=False, scaler=None):
        selections = []
        value = losses["validation"]["loss"]
        if np.isfinite(value) and value < self.best["validation_loss"]:
            self.best["validation_loss"] = float(value)
            selections.append(("best_validation_loss",
                               dict(source="validation", metric="loss", value=float(value))))
        for protocol in MODES:
            if evaluation is None or protocol not in evaluation:
                continue
            value = evaluation[protocol]["oracle"]["f1"]
            key = f"{protocol}_oracle_f1"
            if np.isfinite(value) and value > self.best.get(key, float("-inf")):
                self.best[key] = float(value)
                selections.append((f"best_test_{key}",
                                   dict(source="test", threshold_source="oracle",
                                        procedure=protocol, metric="f1", value=float(value),
                                        threshold=evaluation[protocol]["oracle"]["threshold"])))
        payload = {"model": self.model.state_dict(),
                   "optimizer": self.optimizer.state_dict(),
                   "scheduler": self.scheduler.state_dict(),
                   "epoch": epoch, "epoch_index_base": 1, "next_epoch": epoch,
                   "best": dict(self.best), "best_val_loss": self.best["validation_loss"],
                   "losses": losses, "evaluation": evaluation,
                   "config": dict(self.config), "stage": self.config.get("stage", "pretrain"),
                   **self.extra}
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()
        for stem, selection in selections:
            selected = {**payload, "selection": selection}
            target = self.directory / f"{stem}.pth.tar"
            self._record_save(selected, target)
            if stem == "best_validation_loss":
                if self.config.get("jepa_model"):
                    self._record_save(selected, self.config["jepa_model"])
            if selection.get("procedure") == "timeseries" and self.config.get("jepa_model_best"):
                self._record_save(selected, self.config["jepa_model_best"])
        final = epoch == int(self.config["epochs"])
        scheduled = self.save_every > 0 and epoch % self.save_every == 0
        if evaluated or scheduled or final:
            payload["selection"] = {"source": "epoch", "final": final}
            self._record_save(payload, self.config.get("jepa_checkpoint") or self.directory / "checkpoint.pth.tar")
        if final:
            self._record_save(payload, self.directory / "final.pth.tar")
        (self.directory / "best_metrics.json").write_text(json.dumps(
            {**self.best, "epoch": epoch, "checkpoints": self.saved_checkpoints}, indent=2))
        with (self.directory / "epochs.jsonl").open("a") as stream:
            stream.write(json.dumps({"epoch": epoch, "losses": losses,
                                     "evaluation": evaluation}) + "\n")
        return payload
