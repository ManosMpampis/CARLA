import os
import numpy as np
import pandas as pd
from utils.mypath import MyPath
from .original_dataset import Original_dataset


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

        super().__init__(
            root=root,
            train=train,
            transform=transform,
            sanomaly=sanomaly,
            wsz=wsz,
            stride=stride
        )

        if self.train:
            self.base_folder += "train"
        else:
            self.base_folder += "test"
            self.targets = np.loadtxt(os.path.join(self.root, "test_label", fname)).astype(int)
            # self.targets = np.asarray(self.targets).astype(int)

        file_path = os.path.join(self.root, self.base_folder, fname)
        self.data = pd.read_csv(file_path)
        self.data = np.asarray(self.data).astype(np.float32)

        if np.any(sum(np.isnan(self.data)) != 0):
            print("Data contains NaN which replaced with zero")
            self.data = np.nan_to_num(self.data)

        if self.train:
            self.scaler.fit(self.data)
            # we do not scale the data just yet in order to do it in the Augmented data class.
            # self.data = self.scaler.transform(self.data)

            self.targets = np.zeros(self.data.shape[0]).astype(int)
        else:
            self.scaler.mean_ = mean_data
            self.scaler.scale_ = std_data
            self.data = self.scaler.transform(self.data)
