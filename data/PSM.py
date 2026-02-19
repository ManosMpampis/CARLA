
import os
import pandas
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.mypath import MyPath
import ast
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

device = torch.device("cuda")

class PSM(Dataset):

    base_folder = ''

    def __init__(self, fname, root=MyPath.db_root_dir('psm'), train=True, transform=None, sanomaly= None, mean_data=None, std_data=None, wsz=200, stride=5):

        super(PSM, self).__init__()
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train  # training set or test set
        self.classes = ['Normal', 'Anomaly']

        self.data = []
        self.targets = []
        labels = []

        if self.train:
            self.base_folder += 'train'
        else:
            self.base_folder += 'test'
            labels = pd.read_csv(os.path.join(self.root, 'test_label.csv'))
            labels = labels.drop(columns=['timestamp_(min)'], axis=1)
            labels = np.asarray(labels)

        file_path = os.path.join(self.root, f"{self.base_folder}.csv")
        temp = pd.read_csv(file_path)
        temp = np.asarray(temp)

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

        self.targets = np.asarray(labels)
        self.data = np.asarray(temp)
        self.data, self.targets = self.convert_to_windows(wsz, stride)

    def convert_to_windows(self, w_size, stride):
        windows = []
        wlabels = []
        sz = int((self.data.shape[0]-w_size)/stride)
        for i in range(0, sz):
            st = i * stride
            w = self.data[st:st+w_size]
            if (self.targets[st:st+w_size] > 0).any():                
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
        # ts_org = self.data[index]
        ts_org = torch.from_numpy(self.data[index]).float().to(device)  # cuda

        if len(self.targets) > 0:
            # target = self.targets[index].astype(int)
            target = torch.tensor(self.targets[index].astype(int), dtype=torch.long).to(device)
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