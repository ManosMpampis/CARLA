import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from utils.mypath import MyPath


class WADI(Dataset):
    """`SMD <https://www>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ```` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in a ts
            and returns a transformed version.
    """
    base_folder = ''

    def __init__(self, fname, root=MyPath.db_root_dir('wadi'), train=True, transform=None, sanomaly= None, mean_data=None, std_data=None, device=torch.device("cpu")):

        super(WADI, self).__init__()
        self.device = device
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train  # training set or test set
        self.classes = ['Normal', 'Anomaly']

        self.data = []
        self.targets = []
        labels = []
        wsz, stride = 400, 10

        if self.train:
            fname += '_14days_new.csv'
        else:
            fname += '_attackdataLABLE.csv'

        file_path = os.path.join(self.root, fname)
        fr = pd.read_csv(file_path)

        temp = np.asarray(fr.iloc[:, 3:123])
        # deleted_indeices = np.where((temp == 0) | (temp == ''))[1]
        # temp = np.delete(temp, deleted_indeices, axis=1)

        if np.any(sum(np.isnan(temp))!=0):
            print('Data contains NaN which replaced with zero')
            temp = np.nan_to_num(temp)

        self.mean, self.std = mean_data, std_data
        if self.train:
            self.mean = np.mean(temp, axis=0)
            self.std = np.std(temp , axis=0)
            labels = np.zeros_like(temp)
        else:
            self.std[self.std == 0.0] = 1.0
            temp = (temp - self.mean) / self.std
            labels = np.where((np.asarray(fr['Attack']) == 1), 0, 1)

        self.targets = labels
        self.data = np.asarray(temp)
        self.data, self.targets = self.convert_to_windows(wsz, stride)

    def normalize3(self, a):
        if self.train:
            min_column = np.amin(a, axis=0)
            max_column = np.amax(a, axis=0)
            self.min, self.max = min_column, max_column
        epsilon = 1e-10
        range_column = (self.max - self.min) + epsilon
        normalized_array = (a - self.min) / range_column
        return normalized_array, self.min, self.max

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sam_size = 5
        w_size = w_size * sam_size
        sz = int((self.data.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.data[st:st+w_size:sam_size]
            if self.targets[st:st+w_size:sam_size].any() > 0:
                lbl = 1
            else: lbl=0
            windows.append(w)
            wlabels.append(lbl)
        return np.stack(windows), np.stack(wlabels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        # ts_org = torch.from_numpy(self.data[index]).to(dtype=torch.float32, device=self.device)  # cuda
        ts_org = torch.as_tensor(self.data[index], dtype=torch.float32, device=self.device)
        if len(self.targets) > 0:
            target = torch.tensor(self.targets[index].astype(int), dtype=torch.long, device=self.device)
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''

        ts_size = (ts_org.shape[0], ts_org.shape[1])

        out = {'ts_org': ts_org, 'target': target, 'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}}

        return out

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def get_info(self):
        return self.mean, self.std

    def concat_ds(self, new_ds):
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")