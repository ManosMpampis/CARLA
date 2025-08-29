import os

from torch.utils.data import Dataset
import torch

from utils.utils import mkdir

class SaveAugmentedDataset(Dataset):

    def __init__(self, data, target, fneighbors=None, f_target=None, filename=None):
        super(SaveAugmentedDataset, self).__init__()
        self.classes = ['Normal', 'Anomaly', 'Noise', 'Point', 'Subseq', 'Subseq2']
        self.targets = target
        self.anchors = data

        self.fneighbors = fneighbors
        self.f_target = f_target
        
        self.filename = filename
        if filename is not None:
            self.save_to_file(filename)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        ts = self.anchors[index]
        fneighbor = self.fneighbors[index]
        f_target = self.f_target[index]

        if len(self.targets) > 0:
            target = int(self.targets[index])
            class_name = self.classes[target]
        else:
            target = 0
            class_name = ''

        ts_size = (ts.shape[0])

        out = {'ts_org': ts, 'target': target, 'f_neighbor': fneighbor, 'f_target': f_target,
                'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}}

        return out

    def get_ts(self, index):
        ts = self.anchors[index]
        return ts

    def __len__(self):
        return len(self.anchors)
    
    def save_to_file(self, filename):
        mkdir(os.path.dirname(filename))
        dict = {'data': self.anchors, 'targets': self.targets, 'f_neighbors': self.fneighbors, 'f_target': self.f_target, 
                'classes': self.classes, 'filename': self.filename}
        torch.save(dict, filename)
        print(f"Data saved to {filename}")
    
    def load_from_file(self, filename):
        dict = torch.load(filename)
        self.anchors = dict['data']
        self.targets = dict['targets']
        self.fneighbors = dict['f_neighbors']
        self.f_target = dict['f_target']
        self.classes = dict['classes']
        self.filename = dict['filename']
        print(f"Data loaded from {filename}")

    def get_info(self):
        """Per feature mean and std.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mean, std
        """
        return self.anchors.mean((0, 1)), self.anchors.std((0, 1))
    
    def concat_ds(self, new_ds):
        self.anchors = torch.cat((self.anchors, new_ds.anchors), dim=0)
        self.targets = torch.cat((self.targets, new_ds.targets), dim=0)
        self.fneighbors = torch.cat((self.fneighbors, new_ds.fneighbors), dim=0)
        self.f_target = torch.cat((self.f_target, new_ds.f_target), dim=0)