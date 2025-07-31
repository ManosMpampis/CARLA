import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.distance import euclidean

from utils.utils import log


""" 
    AugmentedDataset
    Returns a ts together with an augmentation.
"""


class AugmentedDataset(Dataset):
    def __init__(self, dataset, preload=True):
        super(AugmentedDataset, self).__init__()
        self.preload = preload
        self.current_epoch = 0
        self.samples = [{} for _ in range(len(dataset))]  # Initialized with empty dictionaries
        transform = dataset.transform
        sanomaly = dataset.sanomaly
        dataset.transform = None
        self.dataset = dataset

        self.mean, self.std = dataset.get_info()

        if isinstance(transform, dict):
            self.ts_transform = transform['standard']
            self.augmentation_transform = transform['augment']
        else:
            self.ts_transform = transform
            self.augmentation_transform = transform
            self.subseq_anomaly = sanomaly
        if preload:
            self.create_pairs()

    def create_pairs(self):
        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        for index in range(len(self.dataset)):
            item = self.dataset.__getitem__(index)
            ts_org = item['ts_org'].clone().detach()
            ts_trg = item['target'].clone().detach()
            
            # Get random neighbor from windows before time step T
            if index > 10:
                rand_nei = np.random.randint(index - 10, index)
                sample_nei = self.dataset.__getitem__(rand_nei)
                ts_w_augment = sample_nei['ts_org'].clone().detach()
            else:
                ts_w_augment = self.augmentation_transform(ts_org)

            ts_ss_augment = self.subseq_anomaly(ts_org)
            if not std.all():
                log('AugmentedDataset: sstd contains zeros')
            std = torch.where(std == 0.0, torch.tensor(1.0), std)

            self.samples[index] = {
                'ts_org': (ts_org - mean) / std,
                'ts_w_augment': (ts_w_augment - mean) / std,
                'ts_ss_augment':  (ts_ss_augment - mean) / std,
                'target': ts_trg
            }

    def __len__(self):
        return len(self.dataset)

    def concat_ds(self, new_ds):
        self.dataset.data = np.concatenate((self.dataset.data, new_ds.dataset.data), axis=0)
        self.dataset.targets = np.concatenate((self.dataset.targets, new_ds.dataset.targets), axis=0)

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_info(self):
        return self.mean, self.std
    
    def __getitem__(self, index):
        if self.preload:
            return self.samples[index]
        item = self.dataset.__getitem__(index)
        ts_org = item['ts_org'].clone().detach()
        ts_trg = item['target'].clone().detach()
        
        # Get random neighbor from windows before time step T
        if index > 10:
            rand_nei = np.random.randint(index - 10, index)
            sample_nei = self.dataset.__getitem__(rand_nei)
            ts_w_augment = sample_nei['ts_org'].clone().detach()
        else:
            ts_w_augment = self.augmentation_transform(ts_org)

        ts_ss_augment = self.subseq_anomaly(ts_org)

        samples = {
            'ts_org': ts_org,
            'ts_w_augment': ts_w_augment,
            'ts_ss_augment':  ts_ss_augment,
            'target': ts_trg,
            'index': index
        }
        return samples
        
class AugmentedDataset_save(Dataset):
    def __init__(self, dataset, preload=True):
        super(AugmentedDataset, self).__init__()
        self.preload = preload
        self.current_epoch = 0
        self.samples = [{} for _ in range(len(dataset))]  # Initialized with empty dictionaries
        transform = dataset.transform
        sanomaly = dataset.sanomaly
        dataset.transform = None
        self.dataset = dataset

        self.mean, self.std = dataset.get_info()

        if isinstance(transform, dict):
            self.ts_transform = transform['standard']
            self.augmentation_transform = transform['augment']
        else:
            self.ts_transform = transform
            self.augmentation_transform = transform
            self.subseq_anomaly = sanomaly
        if preload:
            self.create_pairs()

    def create_pairs(self):
        for index in range(len(self.dataset)):
            item = self.dataset.__getitem__(index)
            ts_org = item['ts_org'].detach()
            ts_trg = item['target'].detach()
            
            # Get random neighbor from windows before time step T
            if index > 10:
                rand_nei = np.random.randint(index - 10, index)
                sample_nei = self.dataset.__getitem__(rand_nei)
                ts_w_augment = sample_nei['ts_org'].detach()
            else:
                ts_w_augment = self.augmentation_transform(ts_org)

            ts_ss_augment = self.subseq_anomaly(ts_org)

            self.samples.append({
                'ts_org': ts_org,
                'ts_w_augment': ts_w_augment,
                'ts_ss_augment':  ts_ss_augment,
                'target': ts_trg
            })

    def __len__(self):
        return len(self.dataset)

    def concat_ds(self, new_ds):
        self.dataset.data = np.concatenate((self.dataset.data, new_ds.dataset.data), axis=0)
        self.dataset.targets = np.concatenate((self.dataset.targets, new_ds.dataset.targets), axis=0)
        self.samples += new_ds.samples

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, index):
        if self.preload:
            return self.samples[index]
        for index in range(len(self.dataset)):
            item = self.dataset.__getitem__(index)
            ts_org = item['ts_org'].clone().detach()
            ts_trg = item['target'].clone().detach()
            
            # Get random neighbor from windows before time step T
            if index > 10:
                rand_nei = np.random.randint(index - 10, index)
                sample_nei = self.dataset.__getitem__(rand_nei)
                ts_w_augment = sample_nei['ts_org'].clone().detach()
            else:
                ts_w_augment = self.augmentation_transform(ts_org)

            ts_ss_augment = self.subseq_anomaly(ts_org)

            samples = {
                'ts_org': ts_org,
                'ts_w_augment': ts_w_augment,
                'ts_ss_augment':  ts_ss_augment,
                'target': ts_trg
            }
            return samples
    
    def save_to_file(self, file_path):
        """
        Save the dataset to a CSV file.
        """
        import pandas as pd
        data = []
        for sample in self.samples:
            data.append({
                'ts_org': sample['ts_org'].numpy(),
                'ts_w_augment': sample['ts_w_augment'].numpy(),
                'ts_ss_augment': sample['ts_ss_augment'].numpy(),
                'target': sample['target'].numpy()
            })
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

""" 
    NeighborsDataset
    Returns a ts with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, transform, N_indices, F_indices, topk):
        super(NeighborsDataset, self).__init__()
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
       
        dataset.transform = None
        self.dataset = dataset

        NN_indices = N_indices.copy() # Nearest neighbor indices (np.array  [len(dataset) x k])
        FN_indices = F_indices.copy()  # Nearest neighbor indices (np.array  [len(dataset) x k])
        if topk is not None:
            print("!!!!!!!!!!!!!!!!!!!!")  # Maybe need to remove the first from nn indices
            self.NN_indices = NN_indices[:, :topk]
            self.FN_indices = FN_indices[:, :topk]  # Was change in making

        num_samples = self.dataset.anchors.shape[0]
        self.NN_index = np.array([np.random.choice(self.NN_indices[i], 1)[0] for i in range(num_samples)])
        self.FN_index = np.array([np.random.choice(self.FN_indices[i], 1)[0] for i in range(num_samples)])
        # NN_index = np.array([np.random.choice(self.NN_indices[i], 1)[0] for i in range(num_samples)])
        # FN_index = np.array([np.random.choice(self.FN_indices[i], 1)[0] for i in range(num_samples)])
        # self.NNeighbor = self.dataset.anchors[NN_index]
        # self.FNeighbor = self.dataset.fneighbors[FN_index]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        

        # NNeighbor = self.NNeighbor.__getitem__(index)
        # FNeighbor = self.FNeighbor.__getitem__(index)
        
        # This can be used to use all top-k neighbors but with one random neighbor every time
        NN_index = np.array([np.random.choice(self.NN_indices[index], 1)[0]])
        FN_index = np.array([np.random.choice(self.FN_indices[index], 1)[0]])
        NNeighbor = self.dataset.__getitem__(self.NN_index[NN_index])['ts_org']
        if hasattr(self.dataset, 'fneighbors'):
            FNeighbor = self.dataset.fneighbors.__getitem__(self.FN_index[FN_index])
        else:
            FNeighbor = self.dataset.__getitem__(self.FN_index[FN_index])['ts_org']  # Just to be backward compatible

        # This can be used to use all top-k neighbors
        # NNeighbor = self.dataset.__getitem__(self.NN_index[index])['ts_org']
        # FNeighbor = self.dataset.fneighbors.__getitem__(self.FN_index[index])

        output['anchor'] = anchor['ts_org']
        output['NNeighbor'] = NNeighbor
        output['FNeighbor'] = FNeighbor
        output['possible_nneighbors'] = torch.as_tensor(self.NN_indices[index])
        output['possible_fneighbors'] = torch.as_tensor(self.FN_indices[index])
        output['target'] = torch.tensor(anchor['target'])
        
        return output

    def concat_ds(self, new_ds):
        self.dataset.data = np.concatenate((self.dataset.data, new_ds.dataset.data), axis=0)
        self.dataset.targets = np.concatenate((self.dataset.targets, new_ds.dataset.targets), axis=0)
