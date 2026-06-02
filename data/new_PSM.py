import os
import numpy as np
import pandas as pd
from utils.mypath import MyPath
from .original_dataset import Original_dataset

class PSM(Original_dataset):
    base_folder = ""

    def __init__(
        self,
        mean_data=None,
        std_data=None,
        root=MyPath.db_root_dir("psm"),
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
            self.targets = pd.read_csv(os.path.join(self.root, "test_label.csv"))
            self.targets = self.targets.drop(columns=["timestamp_(min)"])
            self.targets = np.asarray(self.targets)

        file_path = os.path.join(self.root, f"{self.base_folder}.csv")
        self.data = pd.read_csv(file_path)
        self.data = self.data.drop(columns=["timestamp_(min)"])
        self.data.fillna(
            0, inplace=True
        )  # Replace NaN values with 0 as the original code of RANSynCoders does
        self.data = np.asarray(self.data)

        if np.any(sum(np.isnan(self.data)) != 0):
            print("Data contains NaN which replaced with zero")
            self.data = np.nan_to_num(self.data)

        self.mean, self.std = mean_data, std_data
        if self.train:
            self.scaler.fit(self.data)
            # we do not scale the data just yet in order to do it in the Augmented data class.
            # self.data = self.scaler.transform(self.data)
            self.targets = np.zeros(self.data.shape[0])
        else:
            self.scaler.mean_ = mean_data
            self.scaler.scale_ = std_data
            self.data = self.scaler.transform(self.data)
