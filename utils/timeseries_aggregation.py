"""Map model scores onto timestep or window axes without computing metrics.

All intervals are half-open. ``time_offset`` and ``check_mask`` use original
input timesteps relative to each input window's start. Unlike the legacy
scorer's placement mask, this check_mask actually drops output positions.
"""

import numpy as np
import torch


def _aggregate_none(scores: np.ndarray):
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
    """Mask raw scores, reduce and broadcast if configured, then place check_window.

    All outputs use original timestep indices. A reduced score fills the
    requested interval relative to its input start, regardless of input length.
    """

    def __init__(
            self,
            input_length: int,
            check_window: tuple,
            mask: tuple | None,
            aggregate_scores_strategy: str | None = None,
            weighted_output: bool = False,
            device: str | torch.device | None = "cpu",
            per_window: bool = False
            ):
        if not isinstance(check_window, (tuple, list)) or len(check_window) != 2:
            raise ValueError("check_window must be [start, end]")
        if aggregate_scores_strategy not in AGGREGATE_STRATEGIES:
            raise ValueError("aggregate_scores_strategy must be None, 'mean', or 'sum'")
        if aggregate_scores_strategy is not None:
            check_window = (0, 1) #when average or sum into one value the check window should be only this value
        self.aggregate_scores_strategy = AGGREGATE_STRATEGIES[aggregate_scores_strategy]
        self.weighted_output = weighted_output
        self.device = torch.device(device) if device is not None else None

        self.input_length = input_length
        self.check_window = check_window
        start, end = check_window
        self.width = end - start
        if per_window and self.width == 1:
            raise ValueError("per_window True needs an aggregate_score_strategy or final output width equal to 1")
        self.per_window = per_window
        if not 0 <= start < end:
            raise ValueError("check_window must be a nonempty interval with nonnegative offsets")
        if aggregate_scores_strategy is None and end > self.input_length:
            raise ValueError("Without aggregation, check_window must be within input_length")
        
        self.mask = np.ones(self.width, dtype=np.int8)
        if mask is not None:
            if len(mask) == 2:
                left, right = mask
                if not 0 <= left <= right <= self.width:
                    raise ValueError("mask bounds must be within check_window")
                self.mask.fill(0)
                self.mask[left:right] = 1
            elif np.shape(mask) == (self.width,) and np.isin(mask, [0, 1]).all():
                self.mask = np.array(mask, dtype=np.int8)
            else:
                raise ValueError("mask must be an interval or a binary mask matching check_window")
        self.weights = self.mask if self.weighted_output else (self.mask>0).astype(int)

        self.reset()

    def reset(self):
        self.record = np.empty(0, dtype=float)
        self.counts = np.empty(0, dtype=np.int32)

    def _extract_device(self, model):
        if self.device is None:
            parameters = getattr(model, "parameters", lambda: iter(()))
            self.device = next(iter(parameters()), torch.empty(0)).device

    def _update_batch(self, model, batch):
        ts = batch["ts_org"].to(self.device).transpose(1, 2).contiguous()
        # index_key = "index" if self.per_window else "start_idx" because we check with counts >0 in per_window in evaluation
        self.update_record(model.score(ts), batch["meta"]["start_idx"])

    @torch.no_grad()
    def evaluate(self, model, batch):
        self._extract_device(model)
        self._update_batch(model, batch)
        return self.results()

    __call__ = evaluate

    @torch.no_grad()
    def evaluate_ts(self, model, dataloader):
        self.reset()
        self._extract_device(model)
        for batch in dataloader:
            self._update_batch(model, batch)
        return self.results()

    def update_record(self, scores, starts):
        """Add score rows at their mapped input depending on starts, output check_window, and mask."""

        start, end = self.check_window
        scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else scores.un
        scores = self.aggregate_scores_strategy(scores)[:, start:end] * self.mask
        starts = starts.detach().cpu().numpy() if torch.is_tensor(starts) else starts

        required = int(starts.max()) + self.width
        self._extend_record(required=required)
        for score, window_start in zip(scores, starts):
            window = slice(window_start + start, window_start + end)
            self.record[window] += score
            self.counts[window] += self.weights

    def _extend_record(self, required):
        extra = required - self.record.size
        if extra > 0:
            self.record = np.pad(self.record, (0, extra))
            self.counts = np.pad(self.counts, (0, extra))

    def results(self):
        means = np.full(self.record.shape, 0, dtype=float)
        np.divide(self.record, self.counts, out=means, where=self.counts > 0)
        return means

    def __len__(self):
        return len(self.counts)
