import os

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
        return (self.series.shape[0] - self.wsz) // self.stride + 1


class JEPACorpusDataset(Dataset):
    """Joint-corpus dataset: all SMD machines' train splits in one run.

    Each machine keeps its own per-machine normalization (official-protocol
    compatible); windows are drawn from the concatenation.
    """

    machine_files: list
    targets: np.ndarray

    def __init__(self, p, machine_files: list):
        self.wsz = p["wsz"]
        self.stride = p["stride"]
        root = MyPath.db_root_dir("smd")
        pieces = []
        self.means, self.stds = [], []
        for fname in sorted(machine_files):
            raw = np.asarray(pd.read_csv(os.path.join(root, "train", fname))).astype(np.float32)
            raw = np.nan_to_num(raw)
            scaler = StandardScaler().fit(raw)
            self.means.append(scaler.mean_)
            self.stds.append(scaler.scale_)
            pieces.append(scaler.transform(raw).astype(np.float32))
        lengths = [s.shape[0] for s in pieces]
        self.series = np.concatenate(pieces, axis=0)
        self.offsets = np.cumsum([0] + lengths)
        self.machine_files = sorted(machine_files)

    def __getitem__(self, index):
        start = index * self.stride
        ts = self.series[start:start + self.wsz]
        machine = int(np.searchsorted(self.offsets, start, side="right") - 1)
        meta = {"start_idx": start, "end_idx": start + self.wsz, "index": index,
                "machine": machine}
        return {"ts": ts.astype(np.float32), "meta": meta}

    def __len__(self):
        return (self.series.shape[0] - self.wsz) // self.stride + 1

    @classmethod
    def validation_split(cls, train_dataset):
        """Validation windows from the tail of the pooled train series."""
        val = cls.__new__(cls)
        val.wsz = train_dataset.wsz
        val.stride = train_dataset.stride
        cut = int(train_dataset.series.shape[0] * 0.9)
        val.series = train_dataset.series[cut:]
        val.machine_files = train_dataset.machine_files
        # rebase machine offsets onto the tail so attribution stays correct
        full_offsets = list(train_dataset.offsets)
        rebased = [max(o - cut, 0) for o in full_offsets if o >= cut]
        if not rebased or rebased[0] != 0:
            rebased.insert(0, 0)
        val.offsets = np.asarray(sorted(set(rebased)))
        val.targets = np.zeros(val.series.shape[0], dtype=np.int64)
        return val
