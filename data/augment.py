import numpy as np

class NoiseTransformation(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X):
        """
        Adding random Gaussian noise with mean 0
        """
        noise = np.random.normal(loc=0, scale=self.sigma, size=X.shape).astype(np.float32)  # NumPy operation
        return X + noise
    
class SubAnomaly(object):
    """
    Injects subsequence anomalies into windows.

    Anomaly types:
        0: frequency - repeated/compressed pattern over one long segment
        1: trend     - shifted segment ending at the window edge
        2: spike     - very short high-amplitude burst
        3: scale     - short amplitude-scaled burst
        4: shapelet  - flat plateau segment

    With balanced=True every anomaly type receives the same per-feature
    timestep budget. Short anomaly types (spike, scale) are injected as
    multiple instances at scattered, non-overlapping positions until they
    cover the same number of timesteps as the long anomaly types, so no
    type dominates the data purely because it spans more timesteps. Types
    are also assigned round-robin across windows/features, giving each type
    an equal number of injections overall. `last_injections` records
    (anomaly_type, covered_timesteps) of the most recent call so the
    balance can be audited or logged.
    """

    FREQ = 0
    TREND = 1
    SPIKE = 2
    SCALE = 3
    SHAPELET = 4
    NUM_TYPES = 5
    SHORT_TYPES = (SPIKE, SCALE)

    def __init__(self, portion_len, balanced=True, budget=0.25,
                 spike_len=2, spike_scale=8.0, scale_len=5, scale_scale=3.0,
                 max_instances=32, min_budget=10):
        self.portion_len = portion_len
        self.balanced = balanced
        self.budget = budget
        self.spike_len = spike_len
        self.spike_scale = spike_scale
        self.scale_len = scale_len
        self.scale_scale = scale_scale
        self.max_instances = max_instances
        self.min_budget = min_budget
        self._cycle = 0
        self.last_injections = []

    def inject_frequency_anomaly(self, window,
                                 subsequence_length: int= None,
                                 compression_factor: int = None,
                                 scale_factor: float = None,
                                 trend_factor: float = None,
                                 shapelet_factor: bool = False,
                                 trend_end: bool = False,
                                 start_index: int = None
                                 ):
        """
        Injects an anomaly into a multivariate time series window by manipulating a
        subsequence of the window.

        :param window: The multivariate time series window represented as a 2D tensor.
        :param subsequence_length: The length of the subsequence to manipulate. If None,
                                   the length is chosen randomly between 20% and 90% of
                                   the window length.
        :param compression_factor: The factor by which to compress the subsequence.
                                   If None, the compression factor is randomly chosen
                                   between 2 and 5.
        :param scale_factor: The factor by which to scale the subsequence. If None,
                             the scale factor is chosen randomly between 0.1 and 2.0
                             for each feature in the multivariate series.
        :return: The modified window with the anomaly injected.
        """

        # Clone the input tensor to avoid modifying the original data
        window = window.copy()

        # Set the subsequence_length if not provided
        if subsequence_length is None:
            min_len = int(window.shape[0] * 0.1)
            max_len = int(window.shape[0] * 0.9)
            subsequence_length = np.random.randint(min_len, max_len)

        # Set the compression_factor if not provided
        if compression_factor is None:
            compression_factor = np.random.randint(2, 5)

        # Set the scale_factor if not provided
        if scale_factor is None:
            scale_factor = np.random.uniform(0.1, 2.0, window.shape[1])

        # Randomly select the start index for the subsequence
        if start_index is None:
            start_index = np.random.randint(0, len(window) - subsequence_length)
        end_index = min(start_index + subsequence_length, window.shape[0])

        if trend_end:
            end_index = window.shape[0]

        # Extract the subsequence from the window
        anomalous_subsequence = window[start_index:end_index]

        # Concatenate the subsequence by the compression factor, and then subsample to compress it
        anomalous_subsequence = np.tile(anomalous_subsequence, (compression_factor, 1))
        # anomalous_subsequence = anomalous_subsequence.repeat(compression_factor, 1)  # cuda! PyTorch equivalent of np.tile()
        anomalous_subsequence = anomalous_subsequence[::compression_factor]

        # Scale the subsequence and replace the original subsequence with the anomalous subsequence
        anomalous_subsequence = anomalous_subsequence * scale_factor

        # Trend
        if trend_factor is None:
            trend_factor = np.random.normal(1, 0.5)
        coef = 1
        if np.random.uniform() < 0.5: coef = -1
        anomalous_subsequence = anomalous_subsequence + coef * trend_factor

        if shapelet_factor:
            anomalous_subsequence = window[start_index] + (np.random.rand(len(anomalous_subsequence)) * 0.1).reshape(-1, 1)

        window[start_index:end_index] = anomalous_subsequence

        return np.squeeze(window)

    def _next_type(self):
        anomaly_type = self._cycle % self.NUM_TYPES
        self._cycle += 1
        return anomaly_type

    def _budget_len(self, window_len):
        if isinstance(self.budget, float):
            budget = int(window_len * self.budget)
        else:
            budget = int(self.budget)
        upper = max(1, window_len - 1)
        lower = min(self.min_budget, upper)
        return int(np.clip(budget, lower, upper))

    def _place_segments(self, window_len, num_segments, seg_len, gap=1):
        seg_len = max(1, min(seg_len, window_len // 2))
        starts = []
        candidates = list(range(0, window_len - seg_len + 1))
        np.random.shuffle(candidates)
        for start in candidates:
            if len(starts) >= num_segments:
                break
            if all(start >= s + seg_len + gap or s >= start + seg_len + gap for s in starts):
                starts.append(start)
        if not starts:
            starts = [int(np.random.randint(0, max(1, window_len - seg_len)))]
        return [(s, seg_len) for s in sorted(starts)]

    def _segments_for_type(self, anomaly_type, window_len, budget):
        if anomaly_type in self.SHORT_TYPES:
            seg_len = self.spike_len if anomaly_type == self.SPIKE else self.scale_len
            num_segments = max(1, int(np.ceil(budget / max(1, seg_len))))
            num_segments = min(num_segments, self.max_instances)
            return self._place_segments(window_len, num_segments, seg_len)
        if anomaly_type == self.TREND:
            return [(window_len - budget, budget)]
        start = int(np.random.randint(0, window_len - budget))
        return [(start, budget)]

    def _apply_type(self, column, anomaly_type, segments):
        col = np.asarray(column).reshape(-1, 1)
        for start, seg_len in segments:
            if anomaly_type == self.FREQ:
                col = self.inject_frequency_anomaly(col,
                                                    scale_factor=1,
                                                    trend_factor=0,
                                                    subsequence_length=seg_len,
                                                    start_index=start)
            elif anomaly_type == self.TREND:
                col = self.inject_frequency_anomaly(col,
                                                    compression_factor=1,
                                                    scale_factor=1,
                                                    trend_end=True,
                                                    subsequence_length=seg_len,
                                                    start_index=start)
            elif anomaly_type == self.SPIKE:
                col = self.inject_frequency_anomaly(col,
                                                    subsequence_length=seg_len,
                                                    compression_factor=1,
                                                    scale_factor=self.spike_scale,
                                                    trend_factor=0,
                                                    start_index=start)
            elif anomaly_type == self.SCALE:
                col = self.inject_frequency_anomaly(col,
                                                    subsequence_length=seg_len,
                                                    compression_factor=1,
                                                    scale_factor=self.scale_scale,
                                                    trend_factor=0,
                                                    start_index=start)
            elif anomaly_type == self.SHAPELET:
                col = self.inject_frequency_anomaly(col,
                                                    compression_factor=1,
                                                    scale_factor=1,
                                                    trend_factor=0,
                                                    shapelet_factor=True,
                                                    subsequence_length=seg_len,
                                                    start_index=start)
            col = col.reshape(-1, 1)
        return col.reshape(-1)

    def _balanced_call(self, X):
        anomalous_window = X.copy()
        self.last_injections = []

        if anomalous_window.ndim > 1:
            num_features = anomalous_window.shape[1]
            min_dims = max(1, int(num_features / 10))
            max_dims = max(min_dims + 1, int(num_features / 2))
            num_dims = np.random.randint(min_dims, max_dims)
            feature_ids = np.random.choice(num_features, size=num_dims, replace=False)
            for i in feature_ids:
                anomaly_type = self._next_type()
                column = anomalous_window[:, i]
                budget = self._budget_len(len(column))
                segments = self._segments_for_type(anomaly_type, len(column), budget)
                anomalous_window[:, i] = self._apply_type(column, anomaly_type, segments)
                self.last_injections.append((anomaly_type, sum(l for _, l in segments)))
        else:
            anomaly_type = self._next_type()
            budget = self._budget_len(len(anomalous_window))
            segments = self._segments_for_type(anomaly_type, len(anomalous_window), budget)
            anomalous_window = self._apply_type(anomalous_window, anomaly_type, segments)
            self.last_injections.append((anomaly_type, sum(l for _, l in segments)))

        return anomalous_window

    def _legacy_call(self, X):
        """
        Original random injection: one random-length segment per modified
        feature with a uniformly drawn anomaly type (short types stay short,
        so their timestep coverage is much smaller than the long types).
        """
        anomalous_window = X.copy()

        min_len = int(anomalous_window.shape[0] * 0.1)
        max_len = int(anomalous_window.shape[0] * 0.9)
        subsequence_length = np.random.randint(min_len, max_len)
        start_index = np.random.randint(0, len(anomalous_window) - subsequence_length)
        if (anomalous_window.ndim > 1):
            num_features = anomalous_window.shape[1]
            num_dims = np.random.randint(int(num_features/10), int(num_features/2))
            for _ in range(num_dims):
                i = np.random.randint(0, num_features)
                temp_win = anomalous_window[:, i].reshape((anomalous_window.shape[0], 1))
                anomaly_type = np.random.randint(0, self.NUM_TYPES)
                if anomaly_type == 0:
                    anomalous_window[:, i] = self.inject_frequency_anomaly(temp_win,
                                                                scale_factor=1,
                                                                trend_factor=0,
                                                                subsequence_length=subsequence_length,
                                                                start_index = start_index)
                elif anomaly_type == 1:
                    anomalous_window[:, i] = self.inject_frequency_anomaly(temp_win,
                                                                compression_factor=1,
                                                                scale_factor=1,
                                                                trend_end=True,
                                                                subsequence_length=subsequence_length,
                                                                start_index = start_index)
                elif anomaly_type == 2:
                    anomalous_window[:, i] = self.inject_frequency_anomaly(temp_win,
                                                                subsequence_length=2,
                                                                compression_factor=1,
                                                                scale_factor=8,
                                                                trend_factor=0,
                                                                start_index = start_index)
                elif anomaly_type == 3:
                    anomalous_window[:, i] = self.inject_frequency_anomaly(temp_win,
                                                                subsequence_length=4,
                                                                compression_factor=1,
                                                                scale_factor=3,
                                                                trend_factor=0,
                                                                start_index = start_index)
                elif anomaly_type == 4:
                    anomalous_window[:, i] = self.inject_frequency_anomaly(temp_win,
                                                                compression_factor=1,
                                                                scale_factor=1,
                                                                trend_factor=0,
                                                                shapelet_factor=True,
                                                                subsequence_length=subsequence_length,
                                                                start_index = start_index)

        else:
            temp_win = anomalous_window.reshape((len(anomalous_window), 1))
            anomaly_type = np.random.randint(0, self.NUM_TYPES)
            if anomaly_type == 0:
                anomalous_window = self.inject_frequency_anomaly(temp_win,
                                                                scale_factor=1,
                                                                trend_factor=0,
                                                                subsequence_length=subsequence_length,
                                                                start_index = start_index)
            elif anomaly_type == 1:
                anomalous_window = self.inject_frequency_anomaly(temp_win,
                                                            compression_factor=1,
                                                            scale_factor=1,
                                                            trend_end=True,
                                                            subsequence_length=subsequence_length,
                                                            start_index = start_index)
            elif anomaly_type == 2:
                anomalous_window = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=3,
                                                            compression_factor=1,
                                                            scale_factor=8,
                                                            trend_factor=0,
                                                            start_index = start_index)
            elif anomaly_type == 3:
                anomalous_window = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=5,
                                                            compression_factor=1,
                                                            scale_factor=3,
                                                            trend_factor=0,
                                                            start_index = start_index)
            elif anomaly_type == 4:
                anomalous_window = self.inject_frequency_anomaly(temp_win,
                                                        compression_factor=1,
                                                        scale_factor=1,
                                                        trend_factor=0,
                                                        shapelet_factor=True,
                                                        subsequence_length=subsequence_length,
                                                        start_index = start_index)

        return anomalous_window

    def __call__(self, X):
        """
        Adding sub anomaly with user-defined portion
        """
        if self.balanced:
            return self._balanced_call(X)
        return self._legacy_call(X)
