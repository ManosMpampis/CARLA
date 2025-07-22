import random
import numpy as np
import torch

from utils.utils import log

class NoiseTransformation(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X):
        """
        Adding random Gaussian noise with mean 0
        """
        noise = torch.normal(mean=torch.zeros_like(X, device=X.device), std=torch.ones_like(X, device=X.device) * self.sigma)
        return X + noise

class SubAnomaly(object):
    def __init__(self, portion_len):
        self.portion_len = portion_len
        self.anomalies = ["ANOMALY_SEASONAL", "ANOMALY_TREND", "ANOMALY_GLOBAL", "ANOMALY_CONTEXTUAL", "ANOMALY_SHAPELET"]


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

        # Set the subsequence_length if not provided
        if subsequence_length is None:
            min_len = int(window.shape[0] * 0.1)
            max_len = int(window.shape[0] * 0.9)
            subsequence_length = torch.randint(min_len, max_len, (1,))

        # Set the compression_factor if not provided
        if compression_factor is None:
            compression_factor = torch.randint(2, 5, (1,))

        # Set the scale_factor if not provided
        if scale_factor is None:
            scale_factor = np.random.uniform(0.1, 2.0, window.shape[1])

        # Randomly select the start index for the subsequence
        if start_index is None:
            start_index = torch.randint(0, len(window) - subsequence_length, (1,))
        
        # Calculate the end index for the subsequence
        end_index = window.shape[0] if trend_end else min(start_index + subsequence_length, window.shape[0])

        # Extract the subsequence from the window
        anomalous_subsequence = window[start_index:end_index]

        # Concatenate the subsequence by the compression factor, and then subsample to compress it
        anomalous_subsequence = anomalous_subsequence.repeat(compression_factor, 1)  # cuda! PyTorch equivalent of np.tile()
        anomalous_subsequence = anomalous_subsequence[::compression_factor]

        # Scale the subsequence and replace the original subsequence with the anomalous subsequence
        anomalous_subsequence = anomalous_subsequence * scale_factor

        # Trend
        if trend_factor is None:
            trend_factor = np.random.normal(1, 0.5)
        coef = -1 if (np.random.uniform() < 0.5) else 1
        
        anomalous_subsequence = anomalous_subsequence + coef * trend_factor

        if shapelet_factor:
            anomalous_subsequence = window[start_index] + (torch.rand_like(window[start_index]) * 0.1)  #cuda use!

        window[start_index:end_index] = anomalous_subsequence

        return torch.squeeze(window)

    def __call__(self, X):
        """
        Adding sub anomaly with user-defined portion
        """
        anomalous_window = X.clone() #X.copy()

        min_len = int(anomalous_window.shape[0] * 0.1)
        max_len = int(anomalous_window.shape[0] * 0.9)
        subsequence_length = torch.randint(min_len, max_len, (1,))
        start_index = torch.randint(0, len(anomalous_window) - subsequence_length, (1,))
        if (anomalous_window.ndim > 1):
            num_features = anomalous_window.shape[1]
            random_augmented_features = torch.randint(int(num_features/10), int(num_features/2), (1,)) #(int(num_features/5), int(num_features/2))
            list_non_overlaped_features = random.sample([k for k in range(num_features)], random_augmented_features)
            for augmented_feature in list_non_overlaped_features:
                temp_win = anomalous_window[:, augmented_feature].reshape((anomalous_window.shape[0], 1))
                match random.choice(self.anomalies):
                    case "ANOMALY_SEASONAL":
                        anomalous_window[:, augmented_feature] = self.inject_frequency_anomaly(temp_win,
                                                              scale_factor=1,
                                                              trend_factor=0,
                                                           subsequence_length=subsequence_length,
                                                           start_index = start_index)
                    case "ANOMALY_TREND":
                        anomalous_window[:, augmented_feature] = self.inject_frequency_anomaly(temp_win,
                                                             compression_factor=1,
                                                             scale_factor=1,
                                                             trend_end=True,
                                                           subsequence_length=subsequence_length,
                                                           start_index = start_index)
                    case "ANOMALY_GLOBAL":
                        anomalous_window[:, augmented_feature] = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=2,
                                                            compression_factor=1,
                                                            scale_factor=8,
                                                            trend_factor=0,
                                                           start_index = start_index)
                    case "ANOMALY_CONTEXTUAL":
                        anomalous_window[:, augmented_feature] = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=4,
                                                            compression_factor=1,
                                                            scale_factor=3,
                                                            trend_factor=0,
                                                           start_index = start_index)
                    case "ANOMALY_SHAPELET": 
                        anomalous_window[:, augmented_feature] = self.inject_frequency_anomaly(temp_win,
                                                          compression_factor=1,
                                                          scale_factor=1,
                                                          trend_factor=0,
                                                          shapelet_factor=True,
                                                          subsequence_length=subsequence_length,
                                                          start_index = start_index)
                    case _:
                        log('Anomaly selection error')

        else:
            temp_win = anomalous_window.reshape((len(anomalous_window), 1))
            match random.choice(self.anomalies):
                case "ANOMALY_SEASONAL":
                    anomalous_window = self.inject_frequency_anomaly(temp_win,
                                                                scale_factor=1,
                                                                trend_factor=0,
                                                                subsequence_length=subsequence_length,
                                                                start_index = start_index)
                case "ANOMALY_TREND":
                    anomalous_window = self.inject_frequency_anomaly(temp_win,
                                                         compression_factor=1,
                                                         scale_factor=1,
                                                         trend_end=True,
                                                         subsequence_length=subsequence_length,
                                                         start_index = start_index)
                case "ANOMALY_GLOBAL":
                    anomalous_window = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=3,
                                                        compression_factor=1,
                                                        scale_factor=8,
                                                        trend_factor=0,
                                                        start_index = start_index)
                case "ANOMALY_CONTEXTUAL":
                    anomalous_window = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=5,
                                                        compression_factor=1,
                                                        scale_factor=3,
                                                        trend_factor=0,
                                                        start_index = start_index)
                case "ANOMALY_SHAPELET": 
                    anomalous_window = self.inject_frequency_anomaly(temp_win,
                                                      compression_factor=1,
                                                      scale_factor=1,
                                                      trend_factor=0,
                                                      shapelet_factor=True,
                                                      subsequence_length=subsequence_length,
                                                      start_index = start_index)
                case _:
                        log('Anomaly selection error')

        return anomalous_window







