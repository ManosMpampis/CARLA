"""SubAnomaly mask collator: emits injection mask + injected view.

Unlike InputBlockMaskCollator (mask-only), the steered LeWM needs the actual
SubAnomaly-perturbed values: X_inj, mask = SubAnomaly(X_clean). The injection
span is sampled first so the binary mask matches the perturbed region exactly;
per-channel perturbations reuse data.augment.SubAnomaly.inject_frequency_anomaly
with that fixed (length, start). trend_end extension is disabled so every
channel stays inside the shared temporal mask.
"""
import numpy as np
import torch


class SubAnomalyMaskCollator:
    """Sample one shared temporal span per window and inject SubAnomaly."""

    def __init__(self, min_ratio: float = 0.1, max_ratio: float = 0.9,
                 portion_len=None):
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self._sub = None

    def _subanomaly(self):
        if self._sub is None:
            from data.augment import SubAnomaly

            self._sub = SubAnomaly(portion_len=0.99)
        return self._sub

    def _sample_span(self, window: int):
        lo = max(1, int(window * self.min_ratio))
        hi = max(lo + 1, int(window * self.max_ratio))
        length = int(np.random.randint(lo, hi))
        length = max(1, min(length, window))
        start = int(np.random.randint(0, window - length + 1))
        return length, start

    def _inject_channel(self, temp_win, length: int, start: int):
        sub = self._subanomaly()
        anomaly_type = int(np.random.randint(0, 5))
        if anomaly_type == 0:
            return sub.inject_frequency_anomaly(
                temp_win, scale_factor=1, trend_factor=0,
                subsequence_length=length, start_index=start)
        if anomaly_type == 1:
            return sub.inject_frequency_anomaly(
                temp_win, compression_factor=1, scale_factor=1,
                trend_factor=None, subsequence_length=length,
                start_index=start)
        if anomaly_type == 2:
            return sub.inject_frequency_anomaly(
                temp_win, compression_factor=1, scale_factor=8,
                trend_factor=0, subsequence_length=length,
                start_index=start)
        if anomaly_type == 3:
            return sub.inject_frequency_anomaly(
                temp_win, compression_factor=1, scale_factor=3,
                trend_factor=0, subsequence_length=length,
                start_index=start)
        return sub.inject_frequency_anomaly(
            temp_win, compression_factor=1, scale_factor=1, trend_factor=0,
            shapelet_factor=True, subsequence_length=length,
            start_index=start)

    def __call__(self, batch_size: int, window: int, level_strides,
                 ts=None) -> dict:
        """Return {"input": (B, W) bool, "X_inj": (B, C, W) float}."""
        if ts is None:
            mask = np.zeros((batch_size, window), dtype=bool)
            for b in range(batch_size):
                length, start = self._sample_span(window)
                mask[b, start:start + length] = True
            return {"input": torch.from_numpy(mask)}
        if torch.is_tensor(ts):
            arr = ts.detach().cpu().float().numpy()
        else:
            arr = np.asarray(ts, dtype=np.float32)
        # Accept (B, C, W) model inputs or (B, W, C) dataset windows.
        if arr.ndim == 3 and arr.shape[2] <= arr.shape[1] and window == arr.shape[1]:
            pass  # already (B, C, W) with W second... handled below by shape check
        if arr.ndim != 3:
            raise ValueError(f"expected batched windows, got {arr.shape}")
        b_in, d1, d2 = arr.shape
        if b_in != batch_size:
            batch_size = b_in
        if d2 == window:
            bcw = arr  # (B, C, W)
        elif d1 == window:
            bcw = np.transpose(arr, (0, 2, 1))  # (B, W, C) -> (B, C, W)
        else:
            raise ValueError(f"window {window} matches neither axis of {arr.shape}")
        n_channels = bcw.shape[1]
        mask = np.zeros((batch_size, window), dtype=bool)
        inj = bcw.copy()
        for b in range(batch_size):
            length, start = self._sample_span(window)
            mask[b, start:start + length] = True
            clean_wc = bcw[b].transpose(1, 0).copy()  # (W, C)
            out_wc = clean_wc.copy()
            low = max(1, int(n_channels / 10))
            high = max(low + 1, int(n_channels / 2) + 1)
            num_dims = int(np.random.randint(low, min(high, n_channels + 1)))
            num_dims = max(1, min(num_dims, n_channels))
            for _ in range(num_dims):
                i = int(np.random.randint(0, n_channels))
                temp = out_wc[:, i].reshape((window, 1))
                out_wc[:, i] = np.asarray(
                    self._inject_channel(temp, length, start)).reshape(window)
            inj[b] = out_wc.transpose(1, 0)
        return {"input": torch.from_numpy(mask),
                "X_inj": torch.from_numpy(inj.astype(np.float32))}
