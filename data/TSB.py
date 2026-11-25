import csv
from dataclasses import dataclass
from numbers import Integral
import os
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from utils.mypath import MyPath
from .original_dataset import OriginalDataset


_TRAIN_SUFFIX = re.compile(r"_tr_([0-9]+)_1st_([0-9]+)$")


@dataclass(frozen=True)
class TSBSeries:
    path: Path
    data: np.ndarray
    labels: np.ndarray
    train_end: int
    features: tuple[str, ...]

    @property
    def channels(self) -> int:
        return len(self.features)


def read_tsb_series(path, train_end=None) -> TSBSeries:
    """Read all rows and the declared training boundary without dropping rows.

    ``train_end`` explicitly overrides filename inference, allowing locally
    prepared CSVs while retaining the same last-column ``Label`` contract.
    Both univariate and multivariate data are returned as ``(time, channels)``.
    """
    path = Path(path).expanduser().resolve()
    if train_end is None:
        match = _TRAIN_SUFFIX.search(path.stem)
        if match is None:
            raise ValueError(
                f"{path.name}: cannot infer training prefix; expected filename "
                "ending _tr_<N>_1st_<M>.csv or an explicit train_end"
            )
        train_end = int(match.group(1))
    elif isinstance(train_end, bool) or not isinstance(train_end, Integral):
        raise ValueError(f"{path.name}: train_end must be an integer row count")
    train_end = int(train_end)

    # Inspect the original header before pandas can rename duplicate columns.
    with path.open(newline="", encoding="utf-8-sig") as stream:
        columns = next(csv.reader(stream), [])
    if columns.count("Label") != 1 or not columns or columns[-1] != "Label":
        raise ValueError(
            f"{path.name}: expected exactly one 'Label' column in the final position"
        )
    features = tuple(columns[:-1])
    if not features or any(not name for name in features) or len(set(features)) != len(features):
        raise ValueError(f"{path.name}: feature columns must have nonempty, unique names")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(
                path, encoding="utf-8-sig", skip_blank_lines=False, index_col=False
            )
        data = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
        labels = frame.iloc[:, -1].to_numpy(dtype=np.float32)
    except (ValueError, TypeError, pd.errors.ParserWarning) as exc:
        raise ValueError(
            f"{path.name}: expected one numeric value per feature and Label in every row"
        ) from exc
    if not np.isfinite(data).all():
        raise ValueError(
            f"{path.name}: feature values contain NaN or infinity; repair the CSV "
            "without dropping rows so benchmark labels remain aligned"
        )
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path.name}: Label must contain only binary values 0 and 1")
    if not 0 < train_end < len(data):
        raise ValueError(
            f"{path.name}: train_end={train_end} must be between 1 and "
            f"{len(data) - 1} for {len(data)} rows"
        )
    return TSBSeries(path, data, labels.astype(np.int32), train_end, features)


class TSB(OriginalDataset):
    base_folder = ""

    def __init__(
        self,
        fname,
        mean_data=None,
        std_data=None,
        root=MyPath.db_root_dir("tsb_ad_m"),
        train=True,
        wsz=200,
        stride=5,
        train_end=None,
        evaluation_scope="test",
    ):

        super().__init__(
            root=root,
            train=train,
            wsz=wsz,
            stride=stride
        )
        self.name = "tsb"
        sample = read_tsb_series(os.path.join(root, fname), train_end)

        self.train_end = sample.train_end

        self.evaluation_scope = evaluation_scope
        if self.evaluation_scope not in {"test", "full"}:
            raise ValueError("tsb_evaluation_scope must be test or full")
        
        self.series_start = sample.train_end if self.evaluation_scope == "test" else 0
        
        if self.wsz <= 0 or self.stride <= 0:
            raise ValueError("TSB-AD wsz and stride must both be positive")
        
        if self.train:
            self.data = self.scaler.fit_transform(sample.data[:self.train_end])
            self.targets = sample.labels[:self.train_end]
        else:
            self.scaler.mean_ = mean_data
            self.scaler.scale_ = std_data
        self.extract_test_set(sample)
        if not self.train:
            self.data, self.targets = self.test_data, self.test_target

    def extract_test_set(self, sample):
        self.test_target = sample.labels[self.series_start:]
        self.test_data = self.scaler.transform(sample.data[self.series_start:])

    def _set_training_series(self, val_fraction):
        raw_test_data = self.scaler.inverse_transform(self.test_data)
        super()._set_training_series(val_fraction)
        self.test_data = self.scaler.transform(raw_test_data)

    @classmethod
    def test_split(cls, train_dataset):
        return cls._series_split(train_dataset, train_dataset.test_data, train_dataset.test_target)
