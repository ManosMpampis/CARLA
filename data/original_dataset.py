import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from utils.mypath import MyPath

import torch

device = torch.device("cuda")


class Original_dataset(Dataset):
    base_folder = ""

    def __init__(
        self,
        root="",
        train=True,
        transform=None,
        sanomaly=None,
        wsz=200,
        stride=5,
    ):

        super().__init__()
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train  # training set or test set
        self.classes = ["Normal", "Anomaly"]
        self.wsz = wsz
        self.stride = stride
        self.scaler = StandardScaler()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        idx = index * self.stride
        ts_org = torch.from_numpy(self.data[idx : idx + self.wsz]).float()

        if len(self.targets) > 0:
            # target = self.targets[index].astype(int)
            target = torch.tensor(self.targets[idx], dtype=torch.long)
            class_name = self.classes[target]
        else:
            target = torch.zeros(ts_org.shape[0], dtype=torch.long)
            class_name = ""

        ts_size = (ts_org.shape[0], ts_org.shape[1])

        out = {
            "ts_org": ts_org,
            "target": target,
            "meta": {"start_idx": idx, "end_idx": idx + self.wsz, "ts_size": ts_size, "index": index, "class_name": class_name},
        }

        return out

    def get_ts(self, index):
        idx = index * self.stride
        ts = self.data[idx : idx + self.wsz]
        return ts

    def get_info(self):
        return self.scaler.mean_, self.scaler.scale_

    def concat_ds(self, new_ds):
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def __len__(self):
        return (self.data.shape[0] - self.wsz) // self.stride + 1

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

