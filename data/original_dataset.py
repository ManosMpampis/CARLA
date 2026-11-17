import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

def training_split_bounds(length, window, val_fraction):
    """Use a train-side tail when both splits hold a window, else reuse all rows."""
    if window <= 0:
        raise ValueError("wsz must be positive")
    if not np.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if length < window:
        raise ValueError(f"training series has {length} rows; need at least wsz={window} rows")
    cut = int(length * (1.0 - val_fraction))
    if min(cut, length - cut) < window:
        return length, 0
    return cut, cut


class Original_dataset(Dataset):
    base_folder = ""

    def __init__(
        self,
        root="",
        train=True,
        transform=None,
        sanomaly=None,
        wsz=200,
        stride=5,
    ):

        super().__init__()
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train  # training set or test set
        self.classes = ["Normal", "Anomaly"]
        self.wsz = wsz
        self.stride = stride
        self.scaler = StandardScaler()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'ts': ts, 'target': index of target class, 'meta': dict}
        """
        idx = index * self.stride
        ts_org = self.data[idx : idx + self.wsz]

        if len(self.targets) > 0:
            # target = self.targets[index].astype(int)
            target = self.targets[idx: idx + self.wsz]
            class_name = [self.classes[t] for t in target]
        else:
            target = np.zeros(ts_org.shape[0]).astype(int)
            class_name = [self.classes[t] for t in target]

        ts_size = (ts_org.shape[0], ts_org.shape[1])

        out = {
            "ts_org": ts_org,
            "target": target,
            "meta": {"start_idx": idx, "end_idx": idx + self.wsz, "ts_size": ts_size, "index": index, "class_name": class_name},
        }

        return out

    def _set_training_series(self, val_fraction):
        cut, val_start = training_split_bounds(len(self.data), self.wsz, float(val_fraction))
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        self.data = self.scaler.inverse_transform(self.data)
        
        self.val_data = self.data[val_start:]
        self.val_targets = self.targets[val_start:]

        self.data = self.data[:cut]
        self.targets = self.targets[:cut]
        
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        self.val_data = self.scaler.transform(self.val_data)
        
    @classmethod
    def validation_split(cls, train_dataset, val_fraction):
        """Validation tail, or the full training series when a split is too short."""
        train_dataset._set_training_series(val_fraction=val_fraction)
        val = cls.__new__(cls)
        val.train = False
        val.transform = None
        val.wsz = train_dataset.wsz
        val.stride = train_dataset.stride
        val.data = train_dataset.val_data
        val.targets = train_dataset.val_targets

        val.mean, val.std = train_dataset.mean, train_dataset.std
        return val
    
    def get_ts(self, index):
        idx = index * self.stride
        ts = self.data[idx : idx + self.wsz]
        return ts

    def get_info(self):
        return self.scaler.mean_, self.scaler.scale_

    @property
    def mean(self):
        return self.scaler.mean_

    @property
    def std(self):
        return self.scaler.scale_

    def __len__(self):
        return (self.data.shape[0] - self.wsz) // self.stride + 1

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

