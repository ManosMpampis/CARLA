"""Map model scores onto timestep or window axes without computing metrics.

All intervals are half-open. ``time_offset`` and ``check_mask`` use original
input timesteps relative to each input window's start. Unlike the legacy
scorer's placement mask, this check_mask actually drops output positions.
"""

from numbers import Integral

import numpy as np
import torch


def _integer(name, value, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


class TimeseriesAggregator:
    """Batch/online score aggregation with explicit coverage and label alignment.

    ``model`` is a module with ``score(x)`` or a callable returning scores.
    Inputs are float32 (B,C,input_window); scores must be (B,L). Module score
    dictionaries use their ``fused`` entry. No reconstruction loss is inferred.

    output_fn=None preserves scores and requires L == output_window. ``mean``
    and ``sum`` reduce the raw temporal output, then repeat to output_window.
    Reduction occurs before check_mask cropping.

    On the timestep axis the processed output maps to
    input_start + time_offset + arange(output_window). On the window axis,
    each processed output maps to a successive input-window sample. Its label
    interval is the *raw* model-output interval, intersected with check_mask;
    subsequent output samples advance that interval by stride. Consequently,
    a scalar output can label a checked 50-step region without broadcasting
    its score across 50 timesteps.

    __call__ starts fresh and reads a JEPADataset-compatible DataLoader.
    update accumulates batches; reset(series_length=..., labels=...) initializes
    an online stream. result returns snapshots, never internal mutable state.
    Streaming snapshots do not treat future, not-yet-received outputs as gaps;
    result(finalize=True) validates the complete expected axis.
    """

    def __init__(self, input_window, stride, output_window, model, *,
                 time_offset=0, check_mask=None, output_fn=None, data_fn=None,
                 boundary_policy="zero", gap_policy="error", device=None):
        self.input_window = _integer("input_window", input_window, 1)
        self.stride = _integer("stride", stride, 1)
        self.output_window = _integer("output_window", output_window, 1)
        self.time_offset = _integer("time_offset", time_offset)
        if check_mask is not None:
            if not isinstance(check_mask, (list, tuple)) or len(check_mask) != 2:
                raise ValueError("check_mask must be [start, end]")
            check_mask = tuple(_integer("check_mask boundary", v) for v in check_mask)
            if check_mask[0] >= check_mask[1]:
                raise ValueError("check_mask must be a nonempty half-open interval")
        if output_fn not in (None, "mean", "sum"):
            raise ValueError("output_fn must be None, 'mean', or 'sum'")
        if data_fn not in (None, "window"):
            raise ValueError("data_fn must be None or 'window'")
        if boundary_policy not in ("zero", "crop", "error"):
            raise ValueError("boundary_policy must be 'zero', 'crop', or 'error'")
        if gap_policy not in ("zero", "error"):
            raise ValueError("gap_policy must be 'zero' or 'error'")
        self.model = model
        self.check_mask = check_mask
        self.output_fn = output_fn
        self.data_fn = data_fn
        self.boundary_policy = boundary_policy
        self.gap_policy = gap_policy
        if device is None:
            owner = getattr(model, "__self__", model)
            parameters = getattr(owner, "parameters", lambda: iter(()))()
            device = next(iter(parameters), torch.empty(0)).device
        self.device = torch.device(device)
        self.reset()

    def reset(self, *, series_length=None, labels=None):
        """Clear a stream; optional labels must be binary timestep labels."""
        self.labels = None if labels is None else np.asarray(labels).copy()
        if self.labels is not None:
            if self.labels.ndim != 1 or not np.isin(self.labels, [0, 1]).all():
                raise ValueError("labels must be a one-dimensional binary array")
            self.labels = self.labels.astype(np.int64)
            if series_length is None:
                series_length = len(self.labels)
        self.series_length = (None if series_length is None else
                              _integer("series_length", series_length, 1))
        if self.labels is not None and len(self.labels) != self.series_length:
            raise ValueError("labels must match series_length")
        self._sums = {}
        self._counts = {}
        self._seen = set()
        self._raw_length = None
        return self

    def __call__(self, dataloader):
        """Aggregate one complete loader; require explicit, local start indices."""
        dataset = dataloader.dataset
        if getattr(dataset, "_corpus", None) is not None:
            raise ValueError("Aggregate each series separately, not a multi-series corpus")
        for attribute, expected in (("wsz", self.input_window), ("stride", self.stride)):
            actual = getattr(dataset, attribute, expected)
            if actual != expected:
                raise ValueError(f"Dataset {attribute}={actual} differs from aggregator {expected}")
        if not hasattr(dataset, "series"):
            raise ValueError("dataloader.dataset must expose its original series")
        if getattr(dataloader, "drop_last", False):
            raise ValueError("Aggregation requires drop_last=False")
        self.reset(series_length=len(dataset.series), labels=getattr(dataset, "targets", None))
        for batch in dataloader:
            self.update(batch)
        return self.result(finalize=True)

    def _checked_interval(self, length):
        start, end = self.time_offset, self.time_offset + length
        if self.check_mask is not None:
            start = max(start, self.check_mask[0])
            end = min(end, self.check_mask[1])
        if end <= start:
            raise ValueError("check_mask retains no mapped output positions")
        return start, end

    @torch.no_grad()
    def update(self, batch):
        """Accumulate {'ts': (B,W,C), 'meta': {'start_idx': (B,)}}.

        Starts must follow the configured stride grid. Reordering batches is
        allowed, but duplicate input windows are rejected rather than reweighted.
        """
        windows = torch.as_tensor(batch["ts"], dtype=torch.float32, device=self.device)
        starts = np.asarray(batch["meta"]["start_idx"]).reshape(-1)
        if not np.issubdtype(starts.dtype, np.integer):
            raise ValueError("Input start indices must be integers")
        starts = starts.astype(np.int64)
        if windows.ndim != 3 or windows.shape[1] != self.input_window or len(starts) != len(windows):
            raise ValueError("Batch must contain (B,input_window,C) inputs and B start indices")
        if not len(starts):
            raise ValueError("Cannot update with an empty batch")
        if np.any(starts < 0) or np.any(starts % self.stride):
            raise ValueError("Input starts must be nonnegative and on the stride grid")
        if len(set(starts.tolist())) != len(starts) or any(int(s) in self._seen for s in starts):
            raise ValueError("Duplicate input window start")
        if self.series_length is not None and np.any(starts + self.input_window > self.series_length):
            raise ValueError("Input window extends beyond series_length")
        owner = getattr(self.model, "__self__", self.model)
        training = getattr(owner, "training", None)
        if hasattr(owner, "eval"):
            owner.eval()
        try:
            x = windows.transpose(1, 2).contiguous()
            output = self.model.score(x) if hasattr(self.model, "score") else self.model(x)
        finally:
            if training is not None:
                owner.train(training)
        if isinstance(output, dict):
            output = output["fused"]
        if isinstance(output, torch.Tensor):
            output = output.detach().float().cpu().numpy()
        raw = np.asarray(output, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[0] != len(starts) or raw.shape[1] < 1:
            raise ValueError("Model scores must have shape (B,L), with L >= 1")
        if not np.isfinite(raw).all():
            raise ValueError("Model scores must be finite")
        length = raw.shape[1]
        if self._raw_length is not None and length != self._raw_length:
            raise ValueError("Raw model output length changed during the stream")
        if self.output_fn is None:
            if length != self.output_window:
                raise ValueError("output_fn=None requires raw length == output_window")
            processed = raw
        else:
            reduced = raw.mean(axis=1) if self.output_fn == "mean" else raw.sum(axis=1)
            processed = np.repeat(reduced[:, None], self.output_window, axis=1)
        if not np.isfinite(processed).all():
            raise ValueError("Processed scores must be finite")
        interval = self._checked_interval(length if self.data_fn == "window" else self.output_window)
        self._raw_length = length
        self._seen.update(int(s) for s in starts)
        for row, start in zip(processed, starts):
            start = int(start)
            if self.data_fn == "window":
                for k, value in enumerate(row):
                    position = start // self.stride + k
                    bounds = (start + k * self.stride + interval[0],
                              start + k * self.stride + interval[1])
                    # Window labels retain only the mapped, checked region.
                    if self.series_length is not None and (bounds[1] <= 0 or bounds[0] >= self.series_length):
                        continue
                    if self.series_length is not None and position >= self._axis_length():
                        continue
                    self._add(position, value)
            else:
                for offset in range(interval[0], interval[1]):
                    position = start + offset
                    if position < 0 or (self.series_length is not None and position >= self.series_length):
                        continue
                    self._add(position, row[offset - self.time_offset])
        return self.result()

    def _add(self, position, value):
        self._sums[position] = self._sums.get(position, 0.0) + float(value)
        self._counts[position] = self._counts.get(position, 0) + 1

    def _axis_length(self):
        if self.series_length is None:
            return max(self._counts, default=-1) + 1
        if self.data_fn == "window":
            return max(0, (self.series_length - self.input_window) // self.stride + 1)
        return self.series_length

    def result(self, *, finalize=False):
        """Return scores/labels/indices/counts and checked original intervals.

        Zero filling changes scores only; labels are never overwritten. Crop
        removes matching score/label positions. Interior holes are diagnosed
        even when missing outputs would otherwise appear as ordinary zeros.
        """
        size = self._axis_length()
        indices = np.arange(size, dtype=np.int64)
        counts = np.array([self._counts.get(int(i), 0) for i in indices], dtype=np.int64)
        sums = np.array([self._sums.get(int(i), 0.0) for i in indices], dtype=np.float64)
        scores = np.divide(sums, counts, out=np.zeros(size), where=counts > 0)
        covered = counts > 0
        boundaries = np.zeros(size, dtype=bool)
        if covered.any():
            first, last = np.flatnonzero(covered)[[0, -1]]
            boundaries[:first] = True
            if finalize:
                boundaries[last + 1:] = True
            gaps = ~covered & ~boundaries
            if not finalize:
                gaps[last + 1:] = False
        else:
            boundaries[:] = True
            gaps = np.zeros(size, dtype=bool)
        if finalize and self.gap_policy == "error" and gaps.any():
            raise ValueError(f"Uncovered internal positions: {indices[gaps].tolist()[:10]}")
        if finalize and self.boundary_policy == "error" and boundaries.any():
            raise ValueError("Uncovered boundary positions")
        if finalize and not self._seen:
            raise ValueError("No complete input windows were scored")
        labels = None
        if self.data_fn == "window":
            bounds = []
            for i in indices:
                if self._raw_length is None:
                    bounds.append((int(i) * self.stride, int(i) * self.stride))
                else:
                    a, b = self._checked_interval(self._raw_length)
                    bounds.append((int(i) * self.stride + a, int(i) * self.stride + b))
            intervals = np.asarray(bounds, dtype=np.int64).reshape(-1, 2)
            if self.labels is not None:
                labels = np.array([int(self.labels[max(a, 0):min(b, len(self.labels))].any())
                                   if b > 0 and a < len(self.labels) else 0 for a, b in intervals])
        else:
            intervals = np.column_stack((indices, indices + 1))
            if self.labels is not None:
                labels = self.labels.copy()
        keep = ~boundaries if self.boundary_policy == "crop" else np.ones(size, dtype=bool)
        return dict(scores=scores[keep], labels=None if labels is None else labels[keep],
                    indices=indices[keep], cover_counts=counts[keep],
                    intervals=intervals[keep], axis="window" if self.data_fn else "timestep",
                    uncovered_indices=indices[~covered], finalized=finalize)


def reconstruction_aggregator(model, *, input_window, stride, mode="timeseries", **kwargs):
    """Convenience geometry; all modes still use the same aggregator class.

    For per-timestep non-overlap use mode='non_overlap_timeseries'.
    'non_overlap_window' and 'window' produce one sample per checked region.
    """
    if mode not in {"timeseries", "non_overlap_timeseries", "non_overlap_window", "window"}:
        raise ValueError(f"Unknown reconstruction mode: {mode}")
    window_axis = mode in {"non_overlap_window", "window"}
    if mode.startswith("non_overlap"):
        stride = input_window
    return TimeseriesAggregator(
        input_window=input_window, stride=stride,
        output_window=1 if window_axis else input_window,
        model=model, output_fn="mean" if window_axis else None,
        data_fn="window" if window_axis else None, **kwargs)
