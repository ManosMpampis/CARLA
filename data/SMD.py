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
        self.name = "smd"
        self.fname = fname

        if self.train:
            folder = self.base_folder + "train"
        else:
            folder = self.base_folder + "test"
            self.targets = np.loadtxt(os.path.join(self.root, "test_label", fname)).astype(int)
            # self.targets = np.asarray(self.targets).astype(int)

        self.file_path = os.path.join(self.root, folder, fname)
        self.data = self.extract_data()
        self.features = self.data.shape[-1] #TODO: check if it is correct

        if self.train:
            self.scaler.fit(self.data)
            # we do not scale the data just yet in order to do it in the Augmented data class.
            # self.data = self.scaler.transform(self.data)

            self.targets = np.zeros(self.data.shape[0]).astype(int)
        else:
            self.scaler.mean_ = mean_data
            self.scaler.scale_ = std_data
        self.data = self.scaler.transform(self.data)

    def extract_data(self):
        data = pd.read_csv(self.file_path)
        data = np.asarray(data).astype(np.float32)

        if np.any(sum(np.isnan(data)) != 0):
            print("Data contains NaN which replaced with zero")
            data = np.nan_to_num(data)

    @classmethod
    def test_split(cls, train_dataset):
        """Test tail, or the full training series when a split is too short."""
        test = cls.__new__(cls)
        test.train = False
        test.transform = None
        test.wsz = train_dataset.wsz
        test.stride = train_dataset.stride
        test.scaler = train_dataset.scaler

        test.fname = train_dataset.fname
        folder = train_dataset.base_folder + "test"
        test.file_path = os.path.join(train_dataset.root, folder, train_dataset.fname)
        test.data = test.scalar.transform(test.extract_data())
        test.targets = np.loadtxt(os.path.join(train_dataset.root, "test_label.csv"), skiprows=1, usecols=(1), delimiter=',').astype(int)
        return test
