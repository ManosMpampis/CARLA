import json

import numpy as np
import torch


class Scorer:
    """Emits per-timestep anomaly scores with window bookkeeping.

    ``score_series`` slides windows over a series, computes per-window
    sub-window scores through :meth:`JEPAModel.score`, and aggregates
    overlapping windows overlap-aware: every timestep accumulates one
    contribution per covering window and is divided by its own cover count,
    so points covered by many windows are neither double-counted nor
    dropped. The emitted ``(scores, start_idxs, end_idxs)`` triple is the
    frozen seam consumed by the untouched metrics stack.
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def score_windows(self, windows: torch.Tensor) -> dict:
        """Windows (B, C, W) -> per-window score maps; always fp32."""
        self.model.eval()
        out = self.model.score(windows.float().to(self.device))
        return {
            "fused": out["fused"].float().cpu().numpy(),
            "levels": {k: v.float().cpu().numpy() for k, v in out["levels"].items()},
            "signals": {k: v.float().cpu().numpy() for k, v in out["signals"].items()},
        }

    def score_series(self, series: np.ndarray, wsz: int, stride: int) -> dict:
        """Score a whole (N, C) series.

        Returns aggregated per-timestep arrays (``scores`` plus one entry
        per level/signal in ``channels``) and per-window start/end indices.
        Tail timesteps no full window reaches are forward-filled with the
        last covered value (documented behavior).
        """
        n_steps = series.shape[0]
        starts = list(range(0, max(n_steps - wsz, 0) + 1, stride))
        if n_steps >= wsz and starts[-1] != n_steps - wsz:
            starts.append(n_steps - wsz)
        ends = [s + wsz for s in starts]

        sums = {}
        for begin in range(0, len(starts), 256):
            chunk = starts[begin:begin + 256]
            batch = torch.from_numpy(
                np.stack([series[s:s + wsz] for s in chunk])
            ).permute(0, 2, 1).contiguous()
            result = self.score_windows(batch)

            for name, arr in [("fused", result.pop("fused"))] \
                    + list(result["levels"].items()) \
                    + [(f"signal/{k}", v) for k, v in result["signals"].items()]:
                acc = sums.setdefault(name, np.zeros(n_steps))
                for row, s in zip(arr, chunk):
                    acc[s:s + wsz] += row
        counts = _cover_increments(starts, wsz, n_steps)

        channels = {
            name: _aggregate(acc, counts)
            for name, acc in sums.items()
        }
        scores = channels.pop("fused")
        return {
            "scores": scores,
            "channels": channels,
            "start_idxs": np.array(starts, dtype=np.int64),
            "end_idxs": np.array(ends, dtype=np.int64),
            "cover_counts": counts,
        }


def _cover_increments(starts, wsz, n_steps):
    """How many windows cover each timestep."""
    increments = np.zeros(n_steps, dtype=np.int64)
    for s in starts:
        increments[int(s):int(s) + wsz] += 1
    return increments


def _aggregate(acc: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-window-position sums -> per-timestep means, tail forward-filled
    from the last covered timestep."""
    values = acc / np.maximum(counts, 1)
    covered = int(np.max(np.nonzero(counts)[0])) + 1 if counts.any() else 0
    if 0 < covered < values.shape[0]:
        values[covered:] = values[covered - 1]
    return values


class Calibrator:
    """Train-only threshold calibration.

    Thresholds come exclusively from clean-train score distributions;
    injected-anomaly probe distributions (when supplied) shape only fusion
    weights between already-computed signals. No test labels are reachable
    from this class's inputs by construction. With thin/absent probes the
    fallback is plain mean fusion + clean-train quantile thresholds.
    """

    def __init__(self, quantile: float = 0.995, min_probe_samples: int = 50):
        self.quantile = float(quantile)
        self.min_probe_samples = int(min_probe_samples)
        self.thresholds = {}
        self.weights = {}
        self.fallback = True

    def fit(self, clean: dict, probes: dict = None) -> "Calibrator":
        """clean/probes map signal name -> 1D per-timestep score arrays."""
        clean = dict(clean)
        probes = dict(probes or {})
        for name, values in clean.items():
            values = np.asarray(values, dtype=np.float64)
            self.thresholds[name] = float(np.quantile(values, self.quantile))

        usable = [n for n in clean
                  if n in probes and len(probes[n]) >= self.min_probe_samples]
        separations = {}
        for name in usable:
            c = np.asarray(clean[name], dtype=np.float64)
            p = np.asarray(probes[name], dtype=np.float64)
            sep = (p.mean() - c.mean()) / (c.std() + 1e-9)
            separations[name] = max(sep, 0.0)
        total = sum(separations.values())
        if not usable or total <= 0:
            self.weights = {name: 0.0 for name in clean}
            self.fallback = True
            return self
        self.weights = {name: separations.get(name, 0.0) / total for name in clean}
        self.fallback = False
        return self

    def fuse(self, signals: dict) -> np.ndarray:
        names = list(signals)
        stacked = [np.asarray(signals[n], dtype=np.float64) for n in names]
        if self.fallback or not self.weights:
            return np.mean(stacked, axis=0)
        combined = None
        for n, values in zip(names, stacked):
            term = self.weights.get(n, 0.0) * values
            combined = term if combined is None else combined + term
        return combined

    def threshold_for(self, fused_clean: np.ndarray) -> float:
        return float(np.quantile(np.asarray(fused_clean, dtype=np.float64), self.quantile))

    def save(self, path: str, extra: dict = None) -> None:
        payload = {
            "quantile": self.quantile,
            "thresholds": self.thresholds,
            "weights": self.weights,
            "fallback": bool(self.fallback),
        }
        if extra:
            payload.update(extra)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        with open(path) as f:
            payload = json.load(f)
        calib = cls(quantile=payload.get("quantile", 0.995))
        calib.thresholds = payload.get("thresholds", {})
        calib.weights = payload.get("weights", {})
        calib.fallback = payload.get("fallback", True)
        return calib
