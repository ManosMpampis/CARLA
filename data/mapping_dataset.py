import numpy as np
from torch.utils.data import Dataset
import torch

from utils.timeseries_aggregation import TimeseriesAggregator

# TODO find a way to initialise without the dataset, it is a wrapper
class MappingDataset(Dataset):
    base_folder = ""

    def __init__(
        self,
        dataset,
        latency,
        true_target_policy,
        true_target_window,
        true_target_threshold,
        wsz,
        stride,
        aggregator_config,
    ):

        super().__init__()
        self.dataset = dataset
        self.data = self.dataset.data
        self.targets = self.dataset.targets

        self.original_wsz = dataset.wsz
        self.original_stride = dataset.stride
        self.wsz = wsz if wsz else self.original_wsz
        self.stride = stride if stride else self.original_stride

        self.targets_aggregator = TimeseriesAggregator(aggregator_config)
        self.latency = latency
        # TODO: implement this functionality
        self.true_target_policy = true_target_policy
        self.true_target_window = true_target_window  # training set or test set
        self.true_target_threshold = true_target_threshold
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        idx = index * self.stride
        ts_org = self.data[idx : idx + self.wsz]

        if len(self.targets) > 0:
            target = self.targets[idx+self.latency: idx + self.wsz + self.latency]
        else:
            target = np.zeros(ts_org.shape[0]).astype(int)

        ts_size = (ts_org.shape[0], ts_org.shape[1])

        out = {
            "ts_org": ts_org,
            "target": target,
            "meta": {"start_idx": idx, "end_idx": idx + self.wsz, "ts_size": ts_size, "index": index},
        }
        target_torch = torch.from_numpy(target).unsqueeze() # Aggregator need batches
        idx_torch = torch.Tensor(idx).unsqueeze() # Aggregator need batches
        if len(self.targets_aggregator) != len(self):  #TODO: check if this is true in every case
            self.targets_aggregator.update_record(target_torch, idx_torch)
        return out

    def get_target_ts(self):
        return (self.targets_aggregator.results() >= 1).astype(int)
    
    def reset(self):
        self.targets_aggregator.reset()

    def get_ts(self, index):
        idx = index * self.stride
        ts = self.data[idx : idx + self.wsz]
        return ts

    def get_info(self):
        return self.dataset.mean, self.dataset.std

    @property
    def mean(self):
        return self.dataset.mean

    @property
    def std(self):
        return self.dataset.std

    def __len__(self):
        return (self.data.shape[0] - self.wsz - self.latency) // self.stride + 1

    def extra_repr(self):
        return "Split: Test"
