import os
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.mypath import MyPath


def make_synthetic_series(n_steps: int, n_channels: int, seed: int,
                          with_anomalies: bool = False):
    """Deterministic toy telemetry: smooth periodic components, slow trends
    and light AR noise. With ``with_anomalies`` a few point/spike/level-shift
    segments are injected and a binary label vector is returned."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=np.float64)
    series = np.zeros((n_steps, n_channels), dtype=np.float32)
    for c in range(n_channels):
        f1, f2 = 0.01 + 0.005 * c, 0.03 + 0.002 * c
        series[:, c] = (
            2.0 * np.sin(2 * np.pi * f1 * t + c)
            + 1.0 * np.sin(2 * np.pi * f2 * t + 2 * c)
            + 0.002 * t
        )
    noise = np.zeros((n_steps, n_channels), dtype=np.float32)
    eps = rng.normal(0, 0.05, size=noise.shape).astype(np.float32)
    noise[0] = eps[0]
    for i in range(1, n_steps):
        noise[i] = 0.05 * noise[i - 1] + eps[i]
    series += noise
    labels = np.zeros(n_steps, dtype=np.int64)
    if with_anomalies:
        n_anoms = max(3, n_steps // 500)
        for _ in range(n_anoms):
            kind = rng.integers(0, 3)
            length = int(rng.integers(20, 60))
            start = int(rng.integers(0, max(n_steps - length - 1, 1)))
            channels = rng.choice(n_channels, size=max(1, n_channels // 2), replace=False)
            seg = series[start:start + length]
            if kind == 0:  # amplitude spike
                seg[:, channels] = seg[:, channels] * rng.uniform(4.0, 8.0)
            elif kind == 1:  # level shift
                seg[:, channels] = seg[:, channels] + rng.uniform(3.0, 6.0)
            else:  # frequency break (compressed oscillation)
                idx = np.linspace(0, len(seg) - 1, max(len(seg) // 3, 2)).astype(int)
                series[start:start + len(idx), channels] = \
                    np.asarray(seg)[:, channels][idx] * 3.0
            labels[start:start + length] = 1
    return series.astype(np.float32), labels


class JEPADataset(Dataset):
    """Plain sliding windows over a normalized series.

    Returns {'ts': (wsz, C) float32, 'meta': {start_idx, end_idx, index}}.
    Normalization is fit on the train split only; the optional validation
    split is the tail of the *train* series so checkpoint selection never
    touches test data. Legacy contrastive dataset classes are intentionally
    left untouched and unused (spec decision).
    """

    def __init__(self, p, train: bool, transform=None):
        self.train = train
        self.transform = transform
        self.wsz = p["wsz"]
        self.stride = p["stride"]
        self.seed = p.get("seed", 4)
        source = p["train_db_name"]
        self._corpus = None

        if source == "smd" and str(p.get("fname", "")).lower() in {"all", "*"}:
            if not train:
                raise ValueError(
                    "JEPADataset(fname='all') is a train-corpus dataset; "
                    "score SMD test files separately"
                )
            machine_dir = os.path.join(MyPath.db_root_dir("smd"), "train")
            machines = sorted(
                f for f in os.listdir(machine_dir) if f.startswith("machine-")
            )
            self._corpus = JEPACorpusDataset(p, machines, train=True)
            return

        if source == "synthetic":
            kwargs = dict(p.get("synthetic_kwargs", {}))
            steps = kwargs.get("n_steps", 4000)
            channels = kwargs.get("n_channels", 2)
            if train:
                series, _ = make_synthetic_series(steps, channels, seed=self.seed)
            else:
                test_steps = kwargs.get("test_n_steps", 2000)
                series, self.targets = make_synthetic_series(
                    test_steps, channels, seed=self.seed + 1, with_anomalies=True
                )
            scaler = StandardScaler().fit(series)
            series = scaler.transform(series).astype(np.float32)
            self.mean, self.std = scaler.mean_, scaler.scale_
            if train:
                cut = int(series.shape[0] * (1.0 - p.get("val_fraction", 0.1)))
                self.series = series[:cut]
                self.val_series = series[cut:]
                self.targets = np.zeros(self.series.shape[0], dtype=np.int64)
            else:
                self.series = series
        elif source == "smd":
            fname = p["fname"]
            root = MyPath.db_root_dir("smd")
            if train:
                path = os.path.join(root, "train", fname)
            else:
                path = os.path.join(root, "test", fname)
                self.targets = np.loadtxt(
                    os.path.join(root, "test_label", fname)
                ).astype(int)
            raw = pd.read_csv(path, header=None)
            raw = np.asarray(raw).astype(np.float32)
            raw = np.nan_to_num(raw)
            scaler = StandardScaler()
            train_path = os.path.join(root, "train", fname)
            scaler.fit(np.asarray(pd.read_csv(train_path, header=None)).astype(np.float32))
            self.mean, self.std = scaler.mean_, scaler.scale_
            series = scaler.transform(raw).astype(np.float32)
            if train:
                cut = int(series.shape[0] * (1.0 - p.get("val_fraction", 0.1)))
                self.series = series[:cut]
                self.val_series = series[cut:]
                self.targets = np.zeros(self.series.shape[0], dtype=np.int64)
            else:
                self.series = series
        elif source == "psm":
            from data.PSM import PSM

            legacy_train = PSM(train=True, sanomaly=None, wsz=p["wsz"], stride=p["stride"])
            scaler = StandardScaler().fit(legacy_train.data)
            self.mean, self.std = scaler.mean_, scaler.scale_
            if train:
                series = legacy_train.data
                series = scaler.transform(series).astype(np.float32)
                cut = int(series.shape[0] * (1.0 - p.get("val_fraction", 0.05)))
                self.series = series[:cut]
                self.val_series = series[cut:]
                self.targets = np.zeros(self.series.shape[0], dtype=np.int64)
            else:
                legacy_test = PSM(train=False, sanomaly=None, wsz=p["wsz"],
                                  stride=p["stride"],
                                  mean_data=scaler.mean_, std_data=scaler.scale_)
                self.targets = legacy_test.targets
                self.series = legacy_test.data
        else:
            raise ValueError("Invalid train dataset {}".format(source))

    @classmethod
    def validation_split(cls, train_dataset, p):
        """Validation windows carved out of the train series tail."""
        if getattr(train_dataset, "_corpus", None) is not None:
            return JEPACorpusDataset.validation_split(train_dataset._corpus, p)
        val = cls.__new__(cls)
        val.train = False
        val.transform = None
        val.wsz = train_dataset.wsz
        val.stride = train_dataset.stride
        val.series = train_dataset.val_series
        val.targets = np.zeros(val.series.shape[0], dtype=np.int64)
        val.mean, val.std = train_dataset.mean, train_dataset.std
        return val

    def __getitem__(self, index):
        if self._corpus is not None:
            return self._corpus[index]
        start = index * self.stride
        ts = self.series[start:start + self.wsz]
        meta = {
            "start_idx": start,
            "end_idx": start + self.wsz,
            "index": index,
        }
        out = {"ts": ts.astype(np.float32), "meta": meta}
        if self.transform is not None:
            out["ts"] = self.transform(out["ts"])
        return out

    def __len__(self):
        if self._corpus is not None:
            return len(self._corpus)
        return (self.series.shape[0] - self.wsz) // self.stride + 1


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
        root = MyPath.db_root_dir("smd")
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
            cut = int(length * (1.0 - self.val_fraction))
            if split == "train":
                start, available = 0, cut
            else:
                start, available = cut, length - cut
            self._cuts.append(start)
            counts.append(max((available - self.wsz) // self.stride + 1, 0))
        self._cumulative = np.cumsum([0] + counts, dtype=np.int64)

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
        val.targets = np.zeros(len(val), dtype=np.int64)
        return val
