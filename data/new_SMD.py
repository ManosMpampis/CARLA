import os
import numpy as np
import pandas as pd
from utils.mypath import MyPath
from .original_dataset import Original_dataset

import torch

device = torch.device("cuda")


class SMD(Original_dataset):
    base_folder = ""

    def __init__(
        self,
        fname,
        mean_data=None,
        std_data=None,
        root=MyPath.db_root_dir("smd"),
        train=True,
        transform=None,
        sanomaly=None,
        wsz=200,
        stride=5,
    ):

        super(Original_dataset, self).__init__(
            fname=fname,
            root=root,
            train=train,
            transform=transform,
            sanomaly=sanomaly,
            wsz=wsz,
            stride=stride
        )


        labels = []

        if self.train:
            self.base_folder += "train"
        else:
            self.base_folder += "test"
            labels = pd.read_csv(os.path.join(self.root, "test_label", fname))
            labels = np.asarray(labels)

        file_path = os.path.join(self.root, self.base_folder, fname)
        self.data = pd.read_csv(file_path)
        self.data = np.asarray(self.data)

        if np.any(sum(np.isnan(self.data)) != 0):
            print("Data contains NaN which replaced with zero")
            self.data = np.nan_to_num(self.data)

        if self.train:
            self.scaler.fit(self.data)
            # self.data = self.scaler.transform(self.data)

            labels = np.zeros(self.data.shape[0])
        else:
            self.scaler.mean_ = mean_data
            self.scaler.scale_ = std_data
            self.data = self.scaler.transform(self.data)

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.data.shape[0] - w_size) / stride)
        for i in range(0, sz):
            st = i * stride
            w = self.data[st : st + w_size]
            if (self.targets[st : st + w_size] > 0).any():
                lbl = 1
            else:
                lbl = 0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

    def get_ts(self, index):
        ts = self.data[index]
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

