import os
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.mypath import MyPath
from data.original_dataset import training_split_bounds


class JEPACorpusDataset(Dataset):
    """Memory-bounded joint SMD dataset.

    Each machine is normalized independently and cached as a disk-backed
    ``.npy`` array.  The dataset stores only per-machine metadata in RAM and
    opens cached arrays with ``mmap_mode='r'``.  Windows never cross machine
    boundaries.
    """

    machine_files: list
    targets: np.ndarray

    def __init__(self, p, machine_files: list | None = None, train: bool = True):
        if not train:
            raise ValueError("JEPACorpusDataset currently supports train splits only")
        self.wsz = p["wsz"]
        self.stride = p["stride"]
        self.val_fraction = float(p.get("val_fraction", 0.1))
        root = p.get("dataset_root", MyPath.db_root_dir("smd"))
        cache_dir = p.get("joint_cache_dir",
                          os.path.join(p.get("experiment_dir", "results/smd"),
                                       "joint_cache"))
        os.makedirs(cache_dir, exist_ok=True)
        self.means, self.stds = [], []
        if machine_files is None:
            machine_dir = os.path.join(root, "train")
            machine_files = [f for f in os.listdir(machine_dir)
                             if f.startswith("machine-")]
        self.machine_files = sorted(machine_files)
        self.cache_paths = []
        self.lengths = []
        self.channels = None
        for fname in self.machine_files:
            cache_path = os.path.join(cache_dir, f"{fname}.normalized.npy")
            meta_path = os.path.join(cache_dir, f"{fname}.normalized.json")
            if not os.path.exists(cache_path) or not os.path.exists(meta_path):
                raw = np.asarray(pd.read_csv(
                    os.path.join(root, "train", fname), header=None
                )).astype(np.float32)
                raw = np.nan_to_num(raw)
                scaler = StandardScaler().fit(raw)
                normalized = scaler.transform(raw).astype(np.float32)
                np.save(cache_path, normalized)
                metadata = {"mean": scaler.mean_.tolist(),
                            "std": scaler.scale_.tolist(),
                            "length": int(normalized.shape[0]),
                            "channels": int(normalized.shape[1])}
                with open(meta_path, "w") as stream:
                    json.dump(metadata, stream)
                del raw, normalized
            with open(meta_path) as stream:
                metadata = json.load(stream)
            self.cache_paths.append(cache_path)
            self.lengths.append(int(metadata["length"]))
            self.means.append(np.asarray(metadata["mean"], dtype=np.float64))
            self.stds.append(np.asarray(metadata["std"], dtype=np.float64))
            if self.channels is None:
                self.channels = int(metadata["channels"])
            elif self.channels != int(metadata["channels"]):
                raise ValueError("Joint SMD training requires equal channel counts")
        self._mapped = {}
        self._set_split("train")

    def __getitem__(self, index):
        machine, local_index = self._locate(index)
        cut = self._cuts[machine]
        start = cut + local_index * self.stride
        array = self._mapped_array(machine)
        ts = array[start:start + self.wsz]
        meta = {"start_idx": start, "end_idx": start + self.wsz,
                "index": index, "machine": machine,
                "machine_file": self.machine_files[machine]}
        return {"ts": ts.astype(np.float32), "meta": meta}

    def __len__(self):
        return int(self._cumulative[-1])

    def _set_split(self, split):
        self.split = split
        self._cuts = []
        counts = []
        for length in self.lengths:
            cut, val_start = training_split_bounds(length, self.wsz, self.val_fraction)
            if split == "train":
                start, available = 0, cut
            else:
                start, available = val_start, length - val_start
            self._cuts.append(start)
            counts.append(max((available - self.wsz) // self.stride + 1, 0))
        self._cumulative = np.cumsum([0] + counts, dtype=np.int32)

    def _locate(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        machine = int(np.searchsorted(self._cumulative, index, side="right") - 1)
        return machine, index - int(self._cumulative[machine])

    def _mapped_array(self, machine):
        if machine not in self._mapped:
            self._mapped[machine] = np.load(self.cache_paths[machine], mmap_mode="r")
        return self._mapped[machine]

    @classmethod
    def validation_split(cls, train_dataset, p=None):
        """Validation windows from each machine's train-side tail."""
        val = cls.__new__(cls)
        val.wsz = train_dataset.wsz
        val.stride = train_dataset.stride
        val.val_fraction = train_dataset.val_fraction
        val.machine_files = train_dataset.machine_files
        val.cache_paths = train_dataset.cache_paths
        val.lengths = train_dataset.lengths
        val.channels = train_dataset.channels
        val.means = train_dataset.means
        val.stds = train_dataset.stds
        val._mapped = {}
        val._set_split("val")
        val.targets = np.zeros(len(val), dtype=np.int32)
        return val
