import os
import json
import hashlib
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.mypath import MyPath


def training_split_bounds(length, window, val_fraction):
    """Return a disjoint tail, or an empty validation split for short series."""
    if window <= 0:
        raise ValueError("wsz must be positive")
    if not np.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between 0 and 1")
    if length < window:
        raise ValueError(f"training series has {length} rows; need at least wsz={window} rows")
    if length < 2 * window:
        return length, length
    cut = min(max(int(length * (1.0 - val_fraction)), window), length - window)
    return cut, cut


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


def _numeric_frame(path, *, timestamp=False):
    frame = pd.read_csv(path, header=0 if timestamp else None)
    if timestamp:
        if "timestamp_(min)" not in frame:
            raise ValueError(f"{path}: missing timestamp_(min) column")
        frame = frame.drop(columns=["timestamp_(min)"])
    values = np.nan_to_num(frame.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError(f"{path}: no feature columns")
    features = tuple(str(v) for v in frame.columns) if timestamp else tuple(
        f"channel_{i}" for i in range(values.shape[1]))
    return values, features


def _source_identity(paths):
    return [{"path": str(Path(path).resolve()), "size": Path(path).stat().st_size,
             "mtime_ns": Path(path).stat().st_mtime_ns} for path in paths]


def _raw_source(p):
    source = p["train_db_name"]
    seed = int(p.get("seed", 4))
    if source == "synthetic":
        options = p.get("synthetic_kwargs", {})
        channels = int(options.get("n_channels", 2))
        train, _ = make_synthetic_series(int(options.get("n_steps", 4000)), channels, seed)
        test, labels = make_synthetic_series(int(options.get("test_n_steps", 2000)), channels,
                                             seed + 1, with_anomalies=True)
        return train, test, labels, tuple(f"channel_{i}" for i in range(channels)), {
            "source": "synthetic", "seed": seed, "options": dict(options)}, 0
    if source == "tsb_ad":
        from data.tsb_ad import read_tsb_series
        if not p.get("dataset_root"):
            raise ValueError("TSB-AD requires dataset_root")
        path = Path(p["dataset_root"]) / p["fname"]
        sample = read_tsb_series(path, p.get("tsb_train_end"))
        if p.get("tsb_evaluation_scope", "test") != "test":
            raise ValueError("conditional experiments evaluate the TSB test suffix only")
        return (sample.data[:sample.train_end], sample.data[sample.train_end:],
                sample.labels[sample.train_end:], sample.features,
                {"files": _source_identity([path]), "train_end": sample.train_end,
                 "collection": p.get("dataset_collection", "tsb_ad")}, sample.train_end)
    root = Path(p.get("dataset_root") or MyPath.db_root_dir(source))
    if source == "smd":
        if p.get("fname") in ("all", "all-one", "*"):
            raise ValueError("SMD all is a job selector; all-one requires JEPACorpusDataset")
        paths = [root / kind / p["fname"] for kind in ("train", "test", "test_label")]
        train, features = _numeric_frame(paths[0])
        test, test_features = _numeric_frame(paths[1])
        labels = np.loadtxt(paths[2]).reshape(-1)
    elif source == "psm":
        paths = [root / name for name in ("train.csv", "test.csv", "test_label.csv")]
        train, features = _numeric_frame(paths[0], timestamp=True)
        test, test_features = _numeric_frame(paths[1], timestamp=True)
        label_frame = pd.read_csv(paths[2])
        if label_frame.shape[1] != 2:
            raise ValueError("PSM test_label.csv must contain timestamp and label")
        labels = label_frame.iloc[:, 1].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Invalid train dataset {source}")
    if features != test_features or train.shape[1] != test.shape[1]:
        raise ValueError("training and test feature schemas must match in order")
    if len(labels) != len(test) or not np.isin(labels, (0, 1)).all():
        raise ValueError("test labels must be binary and aligned with test rows")
    return train, test, labels.astype(np.int64), features, {"files": _source_identity(paths)}, 0


class JEPADataset(Dataset):
    """Windows with optimization-row-fitted normalization and disjoint validation.

    full_train_series retains all clean training rows for calibration; a short
    source has no validation windows and uses every training row for optimization.
    Test observations never participate in fitting normalization.
    """

    def __init__(self, p, train: bool, transform=None):
        self.train, self.transform = train, transform
        self.wsz, self.stride = int(p["wsz"]), int(p["stride"])
        if self.stride < 1:
            raise ValueError("stride must be positive")
        self.seed = int(p.get("seed", 4))
        self._corpus = None
        raw, test, labels, schema, identity, self.series_start = _raw_source(p)
        self.features = self.feature_schema = schema
        self.train_end = len(raw)
        cut, val_start = training_split_bounds(len(raw), self.wsz,
                                               float(p.get("val_fraction", 0.1)))
        self.validation_available = val_start < len(raw)
        scaler = StandardScaler().fit(raw[:cut])
        self.mean, self.std = scaler.mean_, scaler.scale_
        self.preprocessing = {"contract": "optimization_rows_standard_scaler_v1",
                              "feature_schema": list(schema), "fit_rows": cut,
                              "mean": self.mean.tolist(), "std": self.std.tolist(),
                              "source": identity}
        normalized = scaler.transform(raw).astype(np.float32)
        self.full_train_series = normalized
        self.val_series = normalized[val_start:]
        if train:
            self.series = normalized[:cut]
            self.targets = np.zeros(len(self.series), dtype=np.int64)
        else:
            self.series = scaler.transform(test).astype(np.float32)
            self.targets = np.asarray(labels, dtype=np.int64)
            if len(self.series) < self.wsz:
                raise ValueError(f"test series has {len(self.series)} rows; need {self.wsz}")

    @classmethod
    def validation_split(cls, train_dataset, p=None):
        val = copy(train_dataset)
        val.train, val.transform = False, None
        val.series = train_dataset.val_series
        val.targets = np.zeros(len(val.series), dtype=np.int64)
        return val

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = int(index) * self.stride
        ts = self.series[start:start + self.wsz].astype(np.float32)
        return {"ts": ts if self.transform is None else self.transform(ts),
                "meta": {"start_idx": start, "end_idx": start + self.wsz,
                         "index": int(index)}}

    def __len__(self):
        return max(0, (len(self.series) - self.wsz) // self.stride + 1)


class JEPACorpusDataset(Dataset):
    """Boundary-safe pooled SMD with fingerprinted, memory-mapped member caches."""

    def __init__(self, p, machine_files=None, train=True):
        if not train:
            raise ValueError("pooled testing evaluates each member independently")
        self.wsz, self.stride = int(p["wsz"]), int(p["stride"])
        self.val_fraction = float(p.get("val_fraction", 0.1))
        self.root = str(Path(p.get("dataset_root") or MyPath.db_root_dir("smd")).resolve())
        self.machine_files = list(machine_files or p.get("machine_files") or sorted(
            path.name for path in (Path(self.root) / "train").glob("machine-*.txt")))
        if not self.machine_files:
            raise ValueError("pooled SMD selection contains no machines")
        cache = Path(p.get("joint_cache_dir") or Path(p["experiment_dir"]) / "joint_cache")
        cache.mkdir(parents=True, exist_ok=True)
        self.cache_paths, self.lengths, self.means, self.stds, self.member_preprocessing = [], [], [], [], []
        self.channels = None
        for name in self.machine_files:
            member = JEPADataset({**p, "fname": name}, train=True)
            if self.channels is None:
                self.channels = len(member.feature_schema)
                self.feature_schema = member.feature_schema
            elif member.feature_schema != self.feature_schema:
                raise ValueError("pooled SMD requires matching feature schemas")
            digest = hashlib.sha256(json.dumps(member.preprocessing, sort_keys=True).encode()).hexdigest()[:20]
            path = cache / f"{name}.{digest}.npy"
            if not path.is_file():
                np.save(path, member.full_train_series)
            self.cache_paths.append(str(path))
            self.lengths.append(len(member.full_train_series))
            self.means.append(member.mean)
            self.stds.append(member.std)
            self.member_preprocessing.append(member.preprocessing)
        self.features = self.feature_schema
        self.preprocessing = {"contract": "pooled_per_machine_v1", "members": dict(
            zip(self.machine_files, self.member_preprocessing)), "feature_schema": list(self.feature_schema)}
        self._mapped = {}
        self._set_split("train")

    def _set_split(self, split):
        self.split = split
        self._cuts, counts = [], []
        self.validation_available = False
        for length in self.lengths:
            cut, val_start = training_split_bounds(length, self.wsz, self.val_fraction)
            self.validation_available |= val_start < length
            start, end = (0, cut) if split == "train" else (val_start, length)
            self._cuts.append(start)
            counts.append(max(0, (end - start - self.wsz) // self.stride + 1))
        self._cumulative = np.cumsum([0] + counts, dtype=np.int64)

    def __len__(self):
        return int(self._cumulative[-1])

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        machine = int(np.searchsorted(self._cumulative, index, side="right") - 1)
        local = int(index) - int(self._cumulative[machine])
        start = self._cuts[machine] + local * self.stride
        if machine not in self._mapped:
            self._mapped[machine] = np.load(self.cache_paths[machine], mmap_mode="r")
        return {"ts": np.asarray(self._mapped[machine][start:start + self.wsz]).astype(np.float32),
                "meta": {"start_idx": start, "end_idx": start + self.wsz,
                         "index": int(index), "machine": machine,
                         "machine_file": self.machine_files[machine]}}

    @classmethod
    def validation_split(cls, train_dataset, p=None):
        val = copy(train_dataset)
        val._mapped = {}
        val._set_split("validation")
        return val
