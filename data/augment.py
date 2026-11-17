import numpy as np
import torch

class SubAnomaly(object):
    def __init__(self, portion_len):
        self.portion_len = portion_len

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

    def __call__(self, X):
        """
        Adding sub anomaly with user-defined portion
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
                anomaly_type = np.random.randint(0, 5)
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
            anomaly_type = np.random.randint(0, 4)
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


class TorchSubAnomaly:
    """Tensor-only counterpart of :class:`SubAnomaly`.

    Accepts floating-point windows shaped (time,) or (time, features).
    Outputs preserve the input device and dtype, and inputs are never mutated.
    Randomness uses PyTorch's RNG, so use ``torch.manual_seed`` to reproduce a
    run; NumPy and PyTorch seeds do not produce identical samples.

    ``portion_len`` is retained for API compatibility. As in SubAnomaly, the
    sampled span uses 10–90% of the window rather than this parameter.
    """

    def __init__(self, portion_len):
        self.portion_len = portion_len

    @staticmethod
    def _validate_window(window, dimensions):
        if not isinstance(window, torch.Tensor):
            raise TypeError("window must be a torch.Tensor")
        if window.ndim not in dimensions or any(size == 0 for size in window.shape):
            raise ValueError("expected a nonempty window shaped (time,) or (time, features)"
                             if 1 in dimensions else
                             "expected a nonempty window shaped (time, features)")
        if not window.is_floating_point():
            raise TypeError("window must have a floating-point dtype")

    @staticmethod
    def _randint(low, high, device):
        return int(torch.randint(low, high, (), device=device).item())

    def _sample_span(self, window):
        time = window.shape[0]
        low = max(1, int(time * 0.1))
        high = max(low + 1, int(time * 0.9))
        length = self._randint(low, high, window.device)
        # Match the original exclusive upper bound, with a full-window fallback
        # for a single time step.
        start = self._randint(0, max(1, time - length), window.device)
        return length, start

    def inject_frequency_anomaly(self, window,
                                 subsequence_length: int = None,
                                 compression_factor: int = None,
                                 scale_factor: float = None,
                                 trend_factor: float = None,
                                 shapelet_factor: bool = False,
                                 trend_end: bool = False,
                                 start_index: int = None):
        """Inject into a (time, features) tensor and squeeze the result.

        Repeats then subsamples the selected span, scales each feature, and
        adds a randomly signed offset. ``trend_end`` extends the span to the
        end of the window. ``shapelet_factor`` replaces these changes with the
        starting value plus shared per-time-step noise in [0, 0.1).
        Factors may also be tensors broadcastable to the selected span.
        """
        self._validate_window(window, (2,))
        device, dtype = window.device, window.dtype
        time, features = window.shape
        if subsequence_length is None:
            low = max(1, int(time * 0.1))
            high = max(low + 1, int(time * 0.9))
            subsequence_length = self._randint(low, high, device)
        if subsequence_length < 1:
            raise ValueError("subsequence_length must be positive")
        if compression_factor is None:
            compression_factor = self._randint(2, 5, device)
        if compression_factor < 1:
            raise ValueError("compression_factor must be positive")
        if scale_factor is None:
            scale_factor = torch.rand(features, device=device, dtype=dtype) * 1.9 + 0.1
        else:
            scale_factor = torch.as_tensor(scale_factor, device=device, dtype=dtype)
        if start_index is None:
            start_index = self._randint(0, max(1, time - subsequence_length), device)
        if not 0 <= start_index < time:
            raise ValueError("start_index must be within the window")
        end_index = time if trend_end else min(start_index + subsequence_length, time)
        subsequence = window[start_index:end_index]
        anomalous = subsequence.repeat(compression_factor, 1)[::compression_factor]
        anomalous = anomalous * scale_factor
        if trend_factor is None:
            trend_factor = torch.randn((), device=device, dtype=dtype) * 0.5 + 1
        else:
            trend_factor = torch.as_tensor(trend_factor, device=device, dtype=dtype)
        sign = torch.where(torch.rand((), device=device) < 0.5, -1, 1)
        anomalous = anomalous + sign * trend_factor
        if shapelet_factor:
            noise = torch.rand((len(subsequence), 1), device=device, dtype=dtype) * 0.1
            anomalous = window[start_index] + noise
        result = window.clone()
        result[start_index:end_index] = anomalous
        return result.squeeze()

    def __call__(self, X):
        """Sample a shared span and perturb channels using the legacy workflow.

        Channels are sampled with replacement. Multivariate windows use all
        five anomaly types; univariate windows use the original four types.
        Windows with fewer than four features always select one channel,
        avoiding the legacy empty range or guaranteed no-op.
        """
        self._validate_window(X, (1, 2))
        length, start = self._sample_span(X)
        univariate = X.ndim == 1
        result = X.clone().reshape(X.shape[0], -1)
        features = result.shape[1]
        if univariate or features < 4:
            num_dims = 1
        else:
            num_dims = self._randint(features // 10, features // 2, X.device)
        for _ in range(num_dims):
            channel = 0 if univariate else self._randint(0, features, X.device)
            anomaly_type = self._randint(0, 4 if univariate else 5, X.device)
            kwargs = dict(subsequence_length=length, start_index=start)
            if anomaly_type == 0:
                kwargs.update(scale_factor=1, trend_factor=0)
            elif anomaly_type == 1:
                kwargs.update(compression_factor=1, scale_factor=1, trend_end=True)
            elif anomaly_type == 2:
                kwargs.update(subsequence_length=3 if univariate else 2,
                              compression_factor=1, scale_factor=8, trend_factor=0)
            elif anomaly_type == 3:
                kwargs.update(subsequence_length=5 if univariate else 4,
                              compression_factor=1, scale_factor=3, trend_factor=0)
            else:
                kwargs.update(compression_factor=1, scale_factor=1,
                              trend_factor=0, shapelet_factor=True)
            # Clone the view so subsequent channel updates do not invalidate
            # values retained by autograd from an earlier injection.
            values = result[:, channel:channel + 1].clone()
            result[:, channel] = self.inject_frequency_anomaly(values, **kwargs).reshape(-1)
        return result.reshape_as(X)
