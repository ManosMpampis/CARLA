import json

import numpy as np
import torch


def evaluation_options(config: dict) -> dict:
    """Resolve evaluation-only YAML options without changing training windows."""
    options = config.get("evaluation", {}) or {}
    if not isinstance(options, dict):
        raise ValueError("evaluation must be a YAML mapping")
    window = int(options.get("input_window", config["wsz"]))
    stride = int(options.get("stride", config["stride"]))
    return {
        "wsz": window,
        "stride": stride,
        "check_mask": options.get("check_mask"),
        "latency_offset": options.get("latency_offset", 0),
        # A final off-grid window would overlap even when stride == window.
        "include_last_window": options.get("include_last_window", stride < window),
    }


def _output_start(wsz: int, output_length: int, check_mask,
                  latency_offset: int) -> int:
    if check_mask is None:
        span = (0, wsz)
    else:
        if not isinstance(check_mask, (list, tuple)) or len(check_mask) != 2:
            raise ValueError("check_mask must be [start, end]")
        span = tuple(check_mask)
    if any(not isinstance(v, (int, np.integer)) for v in span):
        raise ValueError("check_mask boundaries must be integers")
    if span[1] - span[0] != output_length:
        raise ValueError("check_mask span must equal model output length")
    if not isinstance(latency_offset, (int, np.integer)):
        raise ValueError("latency_offset must be an integer")
    return int(span[0] + latency_offset)


def aggregate_score_maps(series: np.ndarray, wsz: int, stride: int,
                         batch_size: int, score_batch, *, check_mask=None,
                         latency_offset: int = 0,
                         include_last_window: bool | None = None) -> dict:
    """Place named (batch, output_length) score maps on the input timeline."""
    n_steps = len(series)
    wsz, stride, batch_size = int(wsz), int(stride), int(batch_size)
    if wsz < 1 or stride < 1 or batch_size < 1:
        raise ValueError("input_window, stride, and batch_size must be positive")
    if n_steps < wsz:
        raise ValueError(f"series length {n_steps} shorter than window {wsz}; "
                         "no full scoring window fits")
    if include_last_window is None:
        include_last_window = stride < wsz
    if not isinstance(include_last_window, bool):
        raise ValueError("include_last_window must be a boolean")
    starts = list(range(0, n_steps - wsz + 1, stride))
    if include_last_window and starts[-1] != n_steps - wsz:
        starts.append(n_steps - wsz)

    sums: dict[str, np.ndarray] = {}
    counts = np.zeros(n_steps, dtype=np.int64)
    output_starts, output_ends, input_starts = [], [], []
    window_scores: dict[str, list[float]] = {}
    output_length = None
    relative_start = None
    names = None
    for begin in range(0, len(starts), batch_size):
        chunk = starts[begin:begin + batch_size]
        windows = np.stack([series[s:s + wsz] for s in chunk])
        maps = score_batch(windows)
        if "fused" not in maps:
            raise ValueError("score_batch must return a fused score map")
        if names is None:
            names = set(maps)
        elif set(maps) != names:
            raise ValueError("score map channels changed between batches")
        if output_length is None:
            fused_shape = np.asarray(maps["fused"]).shape
            if len(fused_shape) != 2 or fused_shape[1] < 1:
                raise ValueError("fused score map must be a nonempty (batch, output_length) array")
            output_length = fused_shape[1]
            relative_start = _output_start(wsz, output_length, check_mask,
                                           latency_offset)
        for name, values in maps.items():
            values = np.asarray(values)
            if values.shape != (len(chunk), output_length):
                raise ValueError(f"score map {name!r} must have shape "
                                 f"({len(chunk)}, {output_length})")
            sums.setdefault(name, np.zeros(n_steps, dtype=np.float64))
        for row_index, s in enumerate(chunk):
            positions = s + relative_start + np.arange(output_length)
            valid = (positions >= 0) & (positions < n_steps)
            if not valid.any():
                continue
            positions = positions[valid]
            offsets = np.flatnonzero(valid)
            counts[positions] += 1
            for name, values in maps.items():
                sums[name][positions] += values[row_index, offsets]
                window_scores.setdefault(name, []).append(
                    float(values[row_index, offsets].mean()))
            output_starts.append(int(positions.min()))
            output_ends.append(int(positions.max()) + 1)
            input_starts.append(s)
    if not output_starts:
        raise ValueError("check_mask and latency_offset place all outputs outside the series")
    return {
        "channels": {name: _aggregate(total, counts) for name, total in sums.items()},
        "start_idxs": np.asarray(output_starts, dtype=np.int64),
        "end_idxs": np.asarray(output_ends, dtype=np.int64),
        "input_start_idxs": np.asarray(input_starts, dtype=np.int64),
        "input_end_idxs": np.asarray(input_starts, dtype=np.int64) + wsz,
        "window_scores": {name: np.asarray(values, dtype=np.float64)
                          for name, values in window_scores.items()},
        "cover_counts": counts,
    }


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

    def __init__(self, model, device, batch_size: int = 256):
        self.model = model.to(device)
        self.device = device
        self.batch_size = max(1, int(batch_size))

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

    def score_series(self, series: np.ndarray, wsz: int, stride: int,
                     batch_size: int | None = None, *, check_mask=None,
                     latency_offset: int = 0,
                     include_last_window: bool | None = None) -> dict:
        """Score a whole (N, C) series.

        Returns aggregated per-timestep arrays (``scores`` plus one entry
        per level/signal in ``channels``) and per-window start/end indices.
        Tail timesteps no full window reaches are forward-filled with the
        last covered value (documented behavior). Windows are forwarded in
        batches of ``batch_size`` (default: the constructor value); every
        start index is scored exactly once regardless of the batching.
        """
        def score_batch(windows):
            batch = torch.from_numpy(windows).permute(0, 2, 1).contiguous()
            return _flatten_score_maps(self.score_windows(batch))

        result = aggregate_score_maps(
            series, wsz, stride, batch_size or self.batch_size, score_batch,
            check_mask=check_mask, latency_offset=latency_offset,
            include_last_window=include_last_window)
        channels = result.pop("channels")
        scores = channels.pop("fused")
        return {"scores": scores, "channels": channels, **result}


def _flatten_score_maps(result: dict) -> dict:
    return {"fused": result["fused"], **result["levels"],
            **{f"signal/{k}": v for k, v in result["signals"].items()}}


class RunningScorer:
    """Incrementally average scores as complete input windows arrive.

    Each update returns the current mean for every timestep touched by that
    window. A later update may change a previously returned mean. Once all
    windows have arrived, the means equal whole-series aggregation.
    """

    def __init__(self, model, device, input_window: int, *, check_mask=None,
                 latency_offset: int = 0, threshold: float | None = None,
                 record_history: bool = False):
        self.scorer = Scorer(model, device)
        self.input_window = int(input_window)
        if self.input_window < 1:
            raise ValueError("input_window must be positive")
        self.check_mask = check_mask
        self.latency_offset = latency_offset
        self.threshold = None if threshold is None else float(threshold)
        if self.threshold is not None and not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        self.record_history = bool(record_history)
        self.counts: dict[int, int] = {}
        self.sums: dict[str, dict[int, float]] = {}
        self.decisions: dict[int, bool] = {}
        self.decision_changes: list[dict] = []
        self.history: list[dict] = []
        self.last_start: int | None = None
        self.output_length: int | None = None
        self.channel_names: set[str] | None = None

    @classmethod
    def from_config(cls, model, device, config: dict, *,
                    threshold: float | None = None,
                    record_history: bool = False) -> "RunningScorer":
        """Use the same placement as the YAML whole-series evaluation."""
        options = evaluation_options(config)
        return cls(model, device, options["wsz"],
                   check_mask=options["check_mask"],
                   latency_offset=options["latency_offset"],
                   threshold=threshold, record_history=record_history)

    def update(self, window: np.ndarray, start_idx: int) -> dict:
        """Score one full (W, C) window and return updated means/counts."""
        window = np.asarray(window, dtype=np.float32)
        if window.ndim != 2 or window.shape[0] != self.input_window:
            raise ValueError("update requires a complete (input_window, channels) window")
        if not isinstance(start_idx, (int, np.integer)) or start_idx < 0:
            raise ValueError("start_idx must be a nonnegative integer")
        if self.last_start is not None and start_idx <= self.last_start:
            raise ValueError("input windows must arrive in increasing start order")
        batch = torch.from_numpy(window[None]).permute(0, 2, 1).contiguous()
        maps = _flatten_score_maps(self.scorer.score_windows(batch))
        fused_shape = np.asarray(maps["fused"]).shape
        if len(fused_shape) != 2 or fused_shape[1] < 1:
            raise ValueError("fused score map must be a nonempty (batch, output_length) array")
        length = fused_shape[1]
        if self.output_length is not None and length != self.output_length:
            raise ValueError("model output length changed between windows")
        if self.channel_names is not None and set(maps) != self.channel_names:
            raise ValueError("score map channels changed between windows")
        relative_start = _output_start(self.input_window, length,
                                       self.check_mask, self.latency_offset)
        for name, values in maps.items():
            if np.asarray(values).shape != (1, length):
                raise ValueError(f"score map {name!r} must have shape (1, {length})")
        positions = int(start_idx) + relative_start + np.arange(length)
        valid = positions >= 0
        positions = positions[valid]
        self.output_length = length
        self.channel_names = set(maps)
        self.last_start = int(start_idx)
        for position in positions:
            index = int(position)
            self.counts[index] = self.counts.get(index, 0) + 1
        current = {}
        for name, values in maps.items():
            totals = self.sums.setdefault(name, {})
            row = np.asarray(values)[0, valid]
            for position, value in zip(positions, row):
                index = int(position)
                totals[index] = totals.get(index, 0.0) + float(value)
            current[name] = np.asarray(
                [totals[int(position)] / self.counts[int(position)]
                 for position in positions], dtype=np.float64)
        decisions = None
        if self.threshold is not None:
            decisions = current["fused"] >= self.threshold
            for position, score, decision in zip(positions,
                                                 current["fused"], decisions):
                index = int(position)
                previous = self.decisions.get(index)
                now = bool(decision)
                if previous is None or previous != now:
                    self.decision_changes.append({
                        "window_start": int(start_idx),
                        "window_end": int(start_idx) + self.input_window,
                        "timestep": index,
                        "previous": previous,
                        "decision": now,
                        "score": float(score),
                        "count": self.counts[index],
                    })
                self.decisions[index] = now
        if self.record_history:
            self.history.append({
                "window_start": int(start_idx),
                "window_end": int(start_idx) + self.input_window,
                "indices": positions.astype(int).tolist(),
                "scores": current["fused"].tolist(),
                "counts": [self.counts[int(position)] for position in positions],
                "decisions": None if decisions is None else decisions.tolist(),
            })
        return {
            "indices": positions.astype(np.int64),
            "channels": current,
            "scores": current["fused"],
            "decisions": decisions,
            "cover_counts": np.asarray(
                [self.counts[int(position)] for position in positions],
                dtype=np.int64),
        }

    def score_at(self, timestep: int, channel: str = "fused") -> float | None:
        """Current mean at one timestep, or None before any score arrives."""
        if timestep not in self.counts:
            return None
        return self.sums[channel][timestep] / self.counts[timestep]

    def decision_at(self, timestep: int) -> bool | None:
        """Latest anomaly decision, or None without a threshold or score."""
        return self.decisions.get(timestep)

    def decision_trace(self) -> dict:
        """Serializable history for inspecting how means and decisions evolved."""
        if not self.record_history:
            raise ValueError("enable record_history to retain a decision trace")
        if self.threshold is None:
            raise ValueError("a threshold is required for decision history")
        if not self.history or not any(update["indices"] for update in self.history):
            raise ValueError("no scored timesteps to display")
        return {"input_window": self.input_window,
                "threshold": self.threshold,
                "updates": self.history,
                "decision_changes": self.decision_changes}

    def save_decision_trace_html(self, path) -> None:
        """Write an interactive replay graph from the recorded history."""
        from utils.decision_trace import save_decision_trace_html
        save_decision_trace_html(self.decision_trace(), path)


def _aggregate(acc: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-window-position sums -> per-timestep means, tail forward-filled
    from the last covered timestep."""
    values = acc / np.maximum(counts, 1)
    covered = int(np.max(np.nonzero(counts)[0])) + 1 if counts.any() else 0
    if 0 < covered < values.shape[0]:
        values[covered:] = values[covered - 1]
    return values


def covered_evaluation_view(scores, labels, starts, ends, cover_counts):
    """Drop unscored positions and remap window intervals for metric inputs."""
    covered = np.asarray(cover_counts) > 0
    if not covered.any():
        raise ValueError("evaluation has no covered timesteps")
    ranks = np.concatenate(([0], np.cumsum(covered)))
    return (np.asarray(scores)[covered], np.asarray(labels)[covered],
            ranks[np.asarray(starts)], ranks[np.asarray(ends)])


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
        self.thresholds: dict[str, float] = {}
        self.weights: dict[str, float] = {}
        self.fallback = True

    def fit(self, clean: dict, probes: dict | None = None) -> "Calibrator":
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
        combined = np.zeros_like(stacked[0])
        for n, values in zip(names, stacked):
            combined = combined + self.weights.get(n, 0.0) * values
        return combined

    def threshold_for(self, fused_clean: np.ndarray) -> float:
        return float(np.quantile(np.asarray(fused_clean, dtype=np.float64), self.quantile))

    def save(self, path: str, extra: dict | None = None) -> None:
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
