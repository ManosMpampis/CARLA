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

def _aggregate_none(scores: np.array):
    return scores

def _aggregate_mean(scores: np.array):
    return np.broadcast_to(scores.mean(axis=1, keepdims=True), scores.shape)

def _aggregate_sum(scores: np.array):
    return np.broadcast_to(scores.sum(axis=1, keepdims=True), scores.shape)

AGGREGATE_STRATEGIES = {
    None: _aggregate_none,
    "mean": _aggregate_mean,
    "sum": _aggregate_sum,
}
    
class TimeseriesAggregator:
    def __init__(
            self,
            input_length: int,
            check_window: tuple, 
            mask: tuple, 
            aggregate_scores_strategy: str = "None", 
            weighted_output: bool = False,
            device: str = "cpu",
            ):
        assert len(check_window) == 2, "check window should have two values, start and end"
        assert aggregate_scores_strategy == "None" or aggregate_scores_strategy == None or aggregate_scores_strategy == "Sum"
        if aggregate_scores_strategy not in (None, "mean", "sum"):
            raise ValueError("aggregate_scores_strategy must be None, 'mean', or 'sum'")
        self.aggregate_scores_strategy = AGGREGATE_STRATEGIES[aggregate_scores_strategy]
        self.weighted_output = weighted_output
        self.device = torch.divide(device)
        # maybe on evaluator
        self.latency = latency
        if data_fn not in (None, "window"):
                    raise ValueError("data_fn must be None or 'window'")

        # Aggregator
        self.check_window = check_window
        self.input_length = input_length
        
        self.mask = np.zeros(input_length, np.int8)
        self.mask = self.mask[self.check_window[0]: self.check_window[1]]
        assert mask == None or len(mask) == 2 or len(mask) == len(self.mask), \
            "Aggregator mask should be of length 2, [start:end] or a complete mask of length equal to your checking output"
        if mask == None:
            self.mask = np.ones_like(self.mask, np.int8)
        if len(mask) == 2:
            for i in range(mask[0], mask[1]):
                self.mask[i] = 1
        elif len(mask) == len(self.mask):
            self.mask = mask
        self.width = len(self.mask)

        self.record = np.empty(0, dtype=float)
        self.counts = np.empty(0, dtype=np.int32)

    def reset(self):
        self.record = np.empty(0, dtype=float)
        self.counts = np.empty(0, dtype=np.int32)

    def _extract_device(self, model):
        if self.device is None:
            device = next(iter(model.parameters), torch.empty(0)).device
            self.device = torch.device(device)

    @torch.no_grad()
    def evaluate(self, model, batch):
        self._extract_device(model)
        ts = batch["ts_org"].to(self.device)
        ts = ts.transpose(1, 2).contiguous()
        starts = batch["meta"]["start_idx"]
        scores = model.score(ts)
        self.update_record(scores, starts)
        return self.results()

    __call__ = evaluate

    @torch.no_grad()
    def evaluate_ts(self, model, dataloader):
        self.reset()
        for batch in dataloader:
            self._extract_device(model)
            ts = batch["ts_org"].to(self.device)
            ts = ts.transpose(1, 2).contiguous()
            starts = batch["meta"]["start_idx"]
            scores = model.score(ts)
            self.update_record(scores, starts)
        return self.results()
    
    def update_record(self, scores, starts):
        scores = self.aggregate_scores_strategy(scores.detach().cpu().numpy())
        starts = starts.detach().cpu().numpy()

        scores = scores[self.check_window[0]: self.check_window[1]]
        scores = scores*self.mask

        if starts.size == 0 or self.width == 0:
            return

        required = int(starts.max()) + self.width
        extra = required - self.record.size
        if extra > 0:
            self.record = np.pad(self.record, (0, extra))
            self.counts = np.pad(self.counts, (0, extra))

        for score, start in zip(scores, starts):
            window = slice(start, start + self.width)
            self.record[window] = score
            self.counts[window] += self.mask if self.weighted_output else 1

        # for score, start in zip(scores, starts):
        #     for i, check in enumerate(range(start,start+len(self.mask))):
        #         if len(self.record)>=check:
        #             self.record[check] = score[i]
        #             self.counts[check] += 1
        #         else:
        #             self.record.append(score[i])
        #             self.counts.append(1)

    def results(self):
        means = np.full(self.record.shape, 0, dtype=float)
        np.divide(self.record, self.counts, out=means, where=self.counts > 0)
        return means

    def __len__(self):
        return self.counts.shape[0] #TODO: check if it is the length

class TimeseriesAggregator2:
    """Batch/online score aggregation with explicit coverage and label alignment.

    ``model`` is a module with ``score(x)`` or a callable returning scores.
    Inputs are (B,C,input_window); scores must be (B,L). Module score
    dictionaries use their ``fused`` entry. No reconstruction loss is inferred.

    output_fn=None preserves scores and requires L == output_window. ``mean``
    and ``sum`` reduce the raw temporal output, then repeat to output_window.
    With output_window=None, use the first score map's length until reset.
    Reduction occurs before check_mask cropping. Window reductions exclude
    scores placed outside series_length.

    On the timestep axis the processed output maps to
    input_start + time_offset + arange(output_window). On the window axis,
    each processed output maps to a successive input-window sample. Its label
    interval is the *raw* model-output interval, intersected with check_mask;
    subsequent output samples advance that interval by stride. Consequently,
    a scalar output can label a checked 50-step region without broadcasting
    its score across 50 timesteps.

    __call__ starts fresh and reads a Original_dataset-compatible DataLoader.
    update accumulates input batches; update_scores accepts precomputed (B,L)
    score maps and input starts without running a model. It returns self without
    materializing a snapshot. reset(series_length=..., labels=...) initializes
    an online stream. result returns snapshots, never internal mutable state.
    Streaming snapshots do not treat future, not-yet-received outputs as gaps;
    result(finalize=True) validates the complete expected axis.
    """

    def __init__(self, input_window, stride, output_window,
                 time_offset=0, check_mask=None, output_fn=None, data_fn=None,
                 boundary_policy="zero", gap_policy="error", device=None,
                 include_last_window=False):
        self.input_window = _integer("input_window", input_window, 1)
        self.stride = _integer("stride", stride, 1)
        self._configured_output_window = (None if output_window is None else
                                          _integer("output_window", output_window, 1))
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
        self.check_mask = check_mask
        self.output_fn = output_fn
        self.data_fn = data_fn
        self.boundary_policy = boundary_policy
        self.gap_policy = gap_policy
        if not isinstance(include_last_window, bool):
            raise ValueError("include_last_window must be a boolean")
        self.include_last_window = include_last_window
        self.device = torch.device(device) if device else None
        self.reset()

    def reset(self, series_length=None, labels=None):
        """Clear a stream; optional labels must be binary timestep labels."""
        self.labels = None if labels is None else np.asarray(labels).copy()
        if self.labels is not None:
            if self.labels.ndim != 1 or not np.isin(self.labels, [0, 1]).all():
                raise ValueError("labels must be a one-dimensional binary array")
            self.labels = self.labels.astype(np.int32)
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
        self.output_window = self._configured_output_window
        return self

    def __call__(self, dataloader, model):
        """Aggregate one complete loader; require explicit, local start indices."""
        dataset = dataloader.dataset
        if getattr(dataset, "_corpus", None) is not None:
            raise ValueError("Aggregate each series separately, not a multi-series corpus")
        for attribute, expected in (("wsz", self.input_window), ("stride", self.stride)):
            actual = getattr(dataset, attribute, expected)
            if actual != expected:
                raise ValueError(f"Dataset {attribute}={actual} differs from aggregator {expected}")
        if getattr(dataloader, "drop_last", False):
            raise ValueError("Aggregation requires drop_last=False")
        self.reset(series_length=len(dataset.data), labels=getattr(dataset, "targets", None))
        for batch in dataloader:
            self.update(batch, model)
        return self.result(finalize=True)

    def _checked_interval(self, length):
        start, end = self.time_offset, self.time_offset + length
        if self.check_mask is not None:
            start = max(start, self.check_mask[0])
            end = min(end, self.check_mask[1])
        if end <= start:
            raise ValueError("check_mask retains no mapped output positions")
        return start, end

    def _extract_device(self, model):
        if self.device is None:
            owner = getattr(model, "__self__", model)
            parameters = getattr(owner, "parameters", lambda: iter(()))()
            device = next(iter(parameters), torch.empty(0)).device
            self.device = torch.device(device)

    @torch.no_grad()
    def update(self, batch, model):
        """Accumulate {'ts': (B,W,C), 'meta': {'start_idx': (B,)}}.

        Reordering is allowed; duplicate windows are rejected. With
        include_last_window=True, the final full off-grid window is allowed.
        """
        self._extract_device(model)
        windows = batch["ts_org"].to(self.device)
        starts = batch["meta"]["start_idx"]
        if windows.ndim != 3 or windows.shape[1] != self.input_window or len(starts) != len(windows):
            raise ValueError("Batch must contain (B,input_window,C) inputs and B start indices")
        x = windows.transpose(1, 2).contiguous()
        output = model.score(x)
        self.update_scores(output, starts)
        return self.result()

    def update_scores(self, scores, starts):
        """Accumulate precomputed (B,L) scores at explicit input-window starts."""
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim != 2 or scores.shape[0] != len(starts) or scores.shape[1] < 1:
            raise ValueError("Model scores must have shape (B,L), with L >= 1")
        length = scores.shape[1]
        if self._raw_length is not None and length != self._raw_length:
            raise ValueError("Raw model output length changed during the stream")
        output_window = length if self.output_window is None else self.output_window
        if self.output_fn is None:
            if length != output_window:
                raise ValueError("output_fn=None requires raw length == output_window")
            processed = scores
        else:
            reduced = scores.mean(axis=1) if self.output_fn == "mean" else scores.sum(axis=1)
            processed = np.repeat(reduced[:, None], output_window, axis=1)
        if not np.isfinite(processed).all():
            raise ValueError("Processed scores must be finite")
        interval = self._checked_interval(length if self.data_fn == "window" else output_window)
        self._raw_length = length
        self.output_window = output_window
        self._seen.update(int(s) for s in starts)
        for row, scores_row, start in zip(processed, scores, starts):
            start = int(start)
            if self.data_fn == "window":
                for k, value in enumerate(row):
                    position = self._window_index(start) + k
                    bounds = (start + k * self.stride + interval[0],
                              start + k * self.stride + interval[1])
                    # Window labels retain only the mapped, checked region.
                    if self.series_length is not None and (bounds[1] <= 0 or bounds[0] >= self.series_length):
                        continue
                    if self.series_length is not None and position >= self._axis_length():
                        continue
                    if self.output_fn is not None and self.series_length is not None:
                        scores_start = start + k * self.stride + self.time_offset
                        left = max(0, -scores_start)
                        right = min(length, self.series_length - scores_start)
                        # Clip the raw map before reducing at series limits.
                        values = scores_row[left:right]
                        if not len(values):
                            continue
                        value = values.mean() if self.output_fn == "mean" else values.sum()
                    self._add(position, value)
            else:
                for offset in range(interval[0], interval[1]):
                    position = start + offset
                    if position < 0 or (self.series_length is not None and position >= self.series_length):
                        continue
                    self._add(position, row[offset - self.time_offset])
        return self

    def _window_index(self, start):
        if start % self.stride:
            return (self.series_length - self.input_window) // self.stride + 1
        return start // self.stride

    def _window_start(self, index):
        if self.series_length is not None:
            return min(index * self.stride, self.series_length - self.input_window)
        return index * self.stride

    def _add(self, position, value):
        self._sums[position] = self._sums.get(position, 0.0) + float(value)
        self._counts[position] = self._counts.get(position, 0) + 1

    def _axis_length(self):
        if self.series_length is None:
            return max(self._counts, default=-1) + 1
        if self.data_fn == "window":
            last = self.series_length - self.input_window
            return max(0, last // self.stride + 1 +
                       int(self.include_last_window and last >= 0 and last % self.stride != 0))
        return self.series_length

    def result(self, finalize=False):
        """Return scores/labels/indices/counts and checked original intervals.

        Zero filling changes scores only; labels are never overwritten. Crop
        removes matching score/label positions. Interior holes are diagnosed
        even when missing outputs would otherwise appear as ordinary zeros.
        """
        size = self._axis_length()
        indices = np.arange(size, dtype=np.int32)
        counts = np.array([self._counts.get(int(i), 0) for i in indices], dtype=np.int32)
        sums = np.array([self._sums.get(int(i), 0.0) for i in indices], dtype=np.float32)
        scores = np.divide(sums, counts, out=np.zeros(size, dtype=np.float32), where=counts > 0)
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
                    bounds.append((self._window_start(int(i)), self._window_start(int(i))))
                else:
                    a, b = self._checked_interval(self._raw_length)
                    bounds.append((self._window_start(int(i)) + a, self._window_start(int(i)) + b))
            intervals = np.asarray(bounds, dtype=np.int32).reshape(-1, 2)
            if self.series_length is not None:
                intervals = np.clip(intervals, 0, self.series_length)
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
