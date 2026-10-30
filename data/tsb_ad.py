"""Strict, alignment-preserving reader for the TSB-AD CSV format.

The declared training prefix comes from ``_tr_N_1st_M.csv``. Labels are
metadata for benchmark evaluation; they never alter that training prefix.
"""

import csv
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd


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
        data = frame.iloc[:, :-1].to_numpy(dtype=np.float64)
        labels = frame.iloc[:, -1].to_numpy(dtype=np.float64)
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
    return TSBSeries(path, data, labels.astype(np.int64), train_end, features)
