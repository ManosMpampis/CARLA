import numpy as np
from torch.utils.data import Dataset
import torch

from utils.timeseries_aggregation import TimeseriesAggregator


class MappingDataset(Dataset):
    """Rewindow a dataset and map its point labels onto the score axis."""

    base_folder = ""

    def __init__(
        self,
        aggregator_config,
        latency: int = 0,
        true_target_policy: str = "count",
        true_target_window: tuple = (0, 1),
        true_target_threshold: float = 1,
        wsz=None,
        stride=None,
    ):

        super().__init__()

        self.eval_wsz = wsz
        self.eval_stride = stride
        self.targets_aggregator = TimeseriesAggregator(**aggregator_config)

        self.latency = int(latency)

        self.true_target_policy = true_target_policy.lower()
        if self.true_target_policy not in ("count", "percent"):
            raise ValueError("true_target_policy must be count or percent")
        self.true_target_window = true_target_window
        if len(true_target_window) != 2 or true_target_window[0] >= true_target_window[1]:
            raise ValueError("true_target_window must be a nonempty [start, end) interval")
        self.true_target_threshold = true_target_threshold

    def assign_dataset(self, dataset):
        self.dataset = dataset
        self.data = self.dataset.data
        self.targets = np.asarray(self.dataset.targets)
        if self.targets.ndim != 1 or len(self.targets) not in (0, len(self.data)):
            raise ValueError("targets must be empty or contain one label per timestep")
        self.original_wsz = dataset.wsz
        self.original_stride = dataset.stride
        self.wsz = self.original_wsz if self.eval_wsz is None else self.eval_wsz
        self.stride = self.original_stride if self.eval_stride is None else self.eval_stride
        self.reset()

    def _target_policy_percent(self, tg_window):
        return ((np.sum(tg_window, axis=0)/len(tg_window))>=self.true_target_threshold).astype(int)
    
    def _target_policy_count(self, tg_window):
        return (np.sum(tg_window, axis=0)>=self.true_target_threshold).astype(int)
    
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
            start = idx + self.latency
            target = self.targets[start : start + self.wsz]
        else:
            target = np.zeros(self.wsz, dtype=int)
        out = {
            "ts_org": ts_org,
            "target": target,
            "meta": {"start_idx": idx, "end_idx": idx + self.wsz, "ts_size": ts_org.shape, "index": index},
        }
        
        starts = index if self.targets_aggregator.per_window else idx
        if len(self.targets_aggregator) != len(self):  #TODO: check if this is true in every case in order to not recalculate the whole timeseries, then save the target_ts one time to use
            self.targets_aggregator.update_record(target[None, ...], starts[None, ...])
        return out

    def get_target_ts(self):
        target_ts = (self.targets_aggregator.results() >= 1).astype(int)
        if self.targets_aggregator.per_window:
            return target_ts[self.targets_aggregator.counts > 0]
        target_ts_range_start = 0
        target_ts_range_end = len(target_ts)
        if self.true_target_window[0] < 0: # Can not check from start so we start further from the start
            target_ts_range_start += self.true_target_window[0]
        if self.true_target_windwo[1] > 1: # Can not check the end of the timeseries so we stop earlier
            target_ts_range_end -= self.true_target_windwo[1]

        for i in range(target_ts_range_start, target_ts_range_end):
            start_idx = i + self.true_target_window[0]
            end_idx = i + self.true_target_window[1]
            target_ts[i] = self._target_policy(target_ts[start_idx : end_idx])
        target_ts = target_ts[target_ts_range_start: target_ts_range_end]
        return target_ts
    
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
