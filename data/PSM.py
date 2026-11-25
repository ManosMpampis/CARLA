import os
import numpy as np
import pandas as pd
from utils.mypath import MyPath
from .original_dataset import OriginalDataset

class PSM(OriginalDataset):
    base_folder = ""

    def __init__(
        self,
        mean_data=None,
        std_data=None,
        root=MyPath.db_root_dir("psm"),
        train=True,
        wsz=200,
        stride=5,
    ):

        super().__init__(
            root=root,
            train=train,
            wsz=wsz,
            stride=stride
        )

        self.name = "psm"
        
        if self.train:
            folder = self.base_folder + "train"
        else:
            folder = self.base_folder + "test"
            self.targets = np.loadtxt(
                os.path.join(self.root, "test_label.csv"),
                skiprows=1, usecols=1, delimiter=",", ndmin=1,
            ).astype(int)

        self.file_path = os.path.join(self.root, f"{folder}.csv")
        self.data = self.extract_data()

        if self.train:
            self.scaler.fit(self.data)
            self.targets = np.zeros(len(self.data), dtype=int)
        else:
            self.scaler.mean_ = mean_data
            self.scaler.scale_ = std_data
        self.data = self.scaler.transform(self.data)

    def extract_data(self):
        data = pd.read_csv(self.file_path)
        return data.drop(columns=["timestamp_(min)"]).fillna(0).to_numpy(dtype=np.float32)

    @classmethod
    def test_split(cls, train_dataset):
        return cls(
            mean_data=train_dataset.mean,
            std_data=train_dataset.std,
            root=train_dataset.root,
            train=False,
            wsz=train_dataset.wsz,
            stride=train_dataset.stride,
        )
