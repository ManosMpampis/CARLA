import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

from utils.mypath import MyPath
from utils.utils import EmptyLogger, Logger, mkdir


class SMD(Dataset):
    def __init__(self, fname, root=MyPath.db_root_dir('smd'), train=True,
                 transform=None, sanomaly= None, mean_data=None, std_data=None,
                 wsz=200, stride=5, logger=None):
        super(SMD, self).__init__()
        self.logger = EmptyLogger() if logger is None else logger
        self.base_folder = ''
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
            labels = pd.read_csv(os.path.join(self.root, 'test_label', fname))
            labels = np.asarray(labels)

        file_path = os.path.join(self.root, self.base_folder, fname)
        temp = pd.read_csv(file_path)
        temp = np.asarray(temp)

        if np.any(sum(np.isnan(temp))!=0):
            self.logger.log('Data contains NaN which replaced with zero')
            temp = np.nan_to_num(temp)

        if isinstance(mean_data, torch.Tensor):
            mean_data = mean_data.cpu().numpy()
            std_data = std_data.cpu().numpy()
        
        self.mean, self.std = mean_data, std_data
        if self.train:
            self.mean = np.mean(temp, axis=0)
            self.std = np.std(temp , axis=0)
            labels = np.zeros_like(temp)
            
            # self.std[self.std == 0.0] = 1.0
            # temp = (temp - self.mean) / self.std
        else:
            if not self.std.all():
                self.logger.log('SMD: sstd contains zeros')
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
        ts_org = torch.as_tensor(self.data[index], dtype=torch.float32)

        if len(self.targets) > 0:
            # target = self.targets[index].astype(int)
            target = torch.tensor(self.targets[index].astype(int), dtype=torch.long)
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

    # def get_info(self):
    #     return self.mean(), self.std()
    
    # @property
    # def mean(self):
    #     if isinstance(self.mean, list):
    #         mean = sum(self.mean)/len(self.mean)
    #     else:
    #         mean = self.mean
    #     return mean
    
    # @property
    # def std(self):
    #     if isinstance(self.mean, list):
    #         std = sum(self.std)/len(self.std)
    #     else:
    #         std = self.std
    #     return std

    def concat_ds(self, new_ds):
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
    

class SMD_all(Dataset):
    def __init__(self, fname, root=MyPath.db_root_dir('smd'), train=True,
                transform=None, sanomaly= None, mean_data=None, std_data=None,
                wsz=200, stride=5, logger=None):

        super(SMD_all, self).__init__()
        self.base_folder = ''
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train  # training set or test set
        self.classes = ['Normal', 'Anomaly']
        self.wsz = wsz
        self.stride = stride
        self.logger = logger if logger is not None else EmptyLogger()
        self.data = []
        self.targets = []
        self.idx = 0
        mean = []
        std = []
        

        if self.train:
            self.base_folder += 'train'
        else:
            self.base_folder += 'test'

        self.folder = os.path.join(self.root, self.base_folder)

        for file_path in os.listdir(self.folder):
            if not file_path.endswith('.txt'):
                continue
            file_path = os.path.join(self.folder, file_path)

            temp = pd.read_csv(file_path)
            temp = np.asarray(temp)

            if np.any(sum(np.isnan(temp))!=0):
                self.logger.log('Data contains NaN which replaced with zero')
                temp = np.nan_to_num(temp)

            if self.train:
                mean.append(np.mean(temp, axis=0))
                std.append(np.std(temp , axis=0))
            else:
                mean.append(mean_data)
                std.append(std_data)
            if not std[0].all():
                self.logger.log('AugmentedDataset: std contains zeros')
            std[0][std[0] == 0.0] = 1.0
        self.mean = sum(mean) / len(mean)
        self.std = sum(std) / len(std)

        for file_path in os.listdir(self.folder):
            if not file_path.endswith('.txt'):
                continue
            file_path = os.path.join(self.folder, file_path)


            temp = pd.read_csv(file_path)
            temp = np.asarray(temp)

            if np.any(sum(np.isnan(temp))!=0):
                self.logger.log('Data contains NaN which replaced with zero')
                temp = np.nan_to_num(temp)

            if self.train:
                labels = np.zeros_like(temp)
            else:
                labels = pd.read_csv(os.path.join(self.root, 'test_label', file_path))
                labels = np.asarray(labels)
            temp = (temp - self.mean) / self.std
            self.targets = np.asarray(labels)
            self.data = np.asarray(temp)
            self.data, self.targets = self.convert_to_windows()
            self.save_to_file(self.folder + '/processed/processed_' + file_path.split('/')[-1].replace('.txt', '.csv'))

    def convert_to_windows(self):
        windows = []
        wlabels = []
        sz = int((self.data.shape[0]-self.wsz)/self.stride)
        for i in range(0, sz):
            st = i * self.stride
            w = self.data[st:st+self.wsz]
            if (self.targets[st:st+self.wsz] > 0).any():
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
        
        ts_org = torch.as_tensor(self.data[index], dtype=torch.float32)

        if len(self.targets) > 0:
            target = torch.tensor(self.targets[index].astype(int), dtype=torch.long)
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''

        ts_size = (ts_org.shape[0], ts_org.shape[1])

        out = {'ts_org': ts_org, 'target': target, 'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}}

        return out

    def save_to_file(self, filename):
        mkdir(os.path.dirname(filename))
        columns=['ts_org', 'target', 'size', 'index', 'class_name']
        df = pd.DataFrame(columns=columns)
        for i in range(len(self.data)):
            out = self.__getitem__(i)  # Ensure data is loaded
            ner_row = pd.DataFrame
            # Save the output to a file or process it as needed
            ts_org = out['ts_org'].cpu().numpy()
            out = [ts_org, out['target'].item(), (ts_org.shape[0], ts_org.shape[1]), self.idx, out['meta']['class_name']]
            ner_row = pd.DataFrame([out], columns=columns)
            df = pd.concat([df, ner_row], ignore_index=True)
            self.idx += 1
        df.to_csv(filename, index=False)

    def get_ts(self, index):
        ts = self.data[index]
        return ts

    def get_info(self):
        return self.mean, self.std

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
    

if __name__ == "__main__":
    file_path = '/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/datasets/SMD'

    logger = Logger("temp", verbose=1, use_tensorboard=False)
    smd = SMD_all(file_path, train=True, logger=logger)
    mean, std = smd.get_info()
    smd = SMD_all(file_path, train=False, logger=logger, mean_data=mean, std_data=std)