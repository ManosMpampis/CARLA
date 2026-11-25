"""Check dataset feature counts, label alignment, and split normalization."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.common_config import get_dataset, get_test_dataset


def check_splits(config, train_rows, test_rows, test_targets, train_targets=None):
    config = {**config, "wsz": 2, "stride": 1, "val_fraction": 0.25}
    train, val = get_dataset(config.copy())
    test = get_test_dataset(config.copy(), train)
    if train_targets is None:
        train_targets = np.zeros(len(train_rows), dtype=int)

    for split, rows, labels in (
        (train, train_rows[:6], train_targets[:6]),
        (val, train_rows[6:], train_targets[6:]),
        (test, test_rows, test_targets),
    ):
        assert split.features == rows.shape[1]
        np.testing.assert_allclose(split.mean, train_rows[:6].mean(axis=0), atol=1e-6)
        np.testing.assert_allclose(split.scaler.inverse_transform(split.data), rows, atol=1e-6)
        np.testing.assert_array_equal(split.targets, labels)
        assert split[0]["ts_org"].shape == (2, rows.shape[1])
        np.testing.assert_array_equal(split[0]["target"], labels[:2])


def main(root):
    train = np.arange(16).reshape(8, 2)
    test = np.arange(16, 24).reshape(4, 2)
    test_labels = np.array([1, 0, 0, 1])

    smd_root = root / "SMD"
    fname = "machine-1-1.txt"
    for folder, rows in (("train", train), ("test", test), ("test_label", test_labels)):
        (smd_root / folder).mkdir(parents=True)
        np.savetxt(smd_root / folder / fname, rows, fmt="%d", delimiter=",")
    check_splits({"name": "smd", "root": smd_root, "fname": fname}, train, test, test_labels)

    psm_root = root / "PSM"
    psm_root.mkdir()
    for name, rows in (("train", train), ("test", test), ("test_label", test_labels[:, None])):
        header = "timestamp_(min),label" if name == "test_label" else "timestamp_(min),a,b"
        np.savetxt(psm_root / f"{name}.csv", np.column_stack((np.arange(len(rows)), rows)),
                   fmt="%d", delimiter=",", header=header, comments="")
    check_splits({"name": "psm", "root": psm_root}, train, test, test_labels)

    train_labels = np.array([0, 0, 1, 0, 0, 0, 0, 1])
    all_labels = np.concatenate((train_labels, test_labels))
    for channels in (1, 2):
        fname = f"tsb_{channels}_tr_8_1st_9.csv"
        all_rows = np.concatenate((train, test))[:, :channels]
        header = "a,Label" if channels == 1 else "a,b,Label"
        np.savetxt(root / fname, np.column_stack((all_rows, all_labels)),
                   fmt="%d", delimiter=",", header=header, comments="")
        config = {"name": "tsb", "root": root, "fname": fname}
        check_splits(config, train[:, :channels], test[:, :channels], test_labels, train_labels)
        check_splits({**config, "evaluation_scope": "full"}, train[:, :channels], all_rows,
                     all_labels, train_labels)


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        main(Path(directory))
    print("SMD, PSM, and TSB dataset splits passed.")
