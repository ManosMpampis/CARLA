"""Verify TSB-AD file contracts and prefix-only normalization without downloads."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.jepa_dataset import JEPADataset
from data.tsb_ad import read_tsb_series


def rejected(action, message):
    try:
        action()
    except ValueError as exc:
        assert message in str(exc), (message, str(exc))
    else:
        raise AssertionError(f"Expected rejection containing {message!r}")


def write_sample(path, data, labels):
    frame = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
    frame["Label"] = labels
    frame.to_csv(path, index=False)


def main():
    with tempfile.TemporaryDirectory(prefix="tsb-ad-data-") as temporary:
        root = Path(temporary)
        path = root / "001_Test_two_channels_tr_64_1st_73.csv"
        steps = np.arange(120, dtype=np.float64)
        data = np.column_stack((steps, steps * 3 + 10))
        labels = np.zeros(len(data), dtype=np.int64)
        labels[73:80] = 1
        write_sample(path, data, labels)
        sample = read_tsb_series(path)
        assert sample.path == path.resolve() and sample.train_end == 64
        assert sample.channels == 2 and sample.features == ("feature_0", "feature_1")
        np.testing.assert_array_equal(sample.data, data)
        np.testing.assert_array_equal(sample.labels, labels)

        config = dict(train_db_name="tsb_ad", dataset_root=str(root), fname=path.name,
                      wsz=8, stride=4, val_fraction=0.25)
        train = JEPADataset(config, train=True)
        val = JEPADataset.validation_split(train, config)
        test = JEPADataset(config, train=False)
        mean, std = data[:64].mean(axis=0), data[:64].std(axis=0)
        expected = ((data - mean) / std).astype(np.float32)
        np.testing.assert_allclose(train.mean, mean)
        np.testing.assert_allclose(train.std, std)
        np.testing.assert_allclose(train.series, expected[:48])
        np.testing.assert_allclose(val.series, expected[48:64])
        np.testing.assert_allclose(test.series, expected)
        np.testing.assert_array_equal(train.targets, np.zeros(48, dtype=int))
        np.testing.assert_array_equal(test.targets, labels)
        assert len(train) == 11 and len(val) == 3 and len(test) == 29
        assert test[18]["meta"] == dict(start_idx=72, end_idx=80, index=18)
        np.testing.assert_allclose(test[18]["ts"], expected[72:80])

        # Neither test values nor labels change normalization or the declared
        # prefix. Some upstream files contain anomalies inside that prefix.
        changed = data.copy()
        changed[64:] *= 1000
        changed_labels = 1 - labels
        write_sample(path, changed, changed_labels)
        again = JEPADataset(config, train=True)
        np.testing.assert_array_equal(again.series, train.series)
        np.testing.assert_array_equal(again.val_series, train.val_series)
        np.testing.assert_array_equal(again.mean, train.mean)
        np.testing.assert_array_equal(again.std, train.std)
        np.testing.assert_array_equal(again.targets, train.targets)
        assert read_tsb_series(path).train_end == 64

        # Univariate files retain their channel axis. Explicit boundaries allow
        # custom local filenames but remain integer, in-range row counts.
        custom = root / "custom.csv"
        write_sample(custom, data[:, :1], labels)
        uni = read_tsb_series(custom, train_end=64)
        assert uni.data.shape == (120, 1) and uni.channels == 1
        rejected(lambda: read_tsb_series(custom), "cannot infer training prefix")
        for boundary in (0, -1, 120, 121):
            rejected(lambda: read_tsb_series(custom, boundary), "must be between")
        for boundary in (True, 1.5, "64"):
            rejected(lambda: read_tsb_series(custom, boundary), "must be an integer")
        ambiguous = root / "001_Test_64_73_80.csv"
        write_sample(ambiguous, data, labels)
        rejected(lambda: read_tsb_series(ambiguous), "cannot infer training prefix")

        rejected(lambda: JEPADataset(dict(config, wsz=17), train=True), "both splits need")
        rejected(lambda: JEPADataset(dict(config, val_fraction=0), train=True), "val_fraction")
        rejected(lambda: JEPADataset(dict(config, stride=0), train=True), "wsz and stride")
        explicit = JEPADataset(dict(config, fname=custom.name, tsb_train_end=64), train=False)
        assert explicit.series.shape == (120, 1)

        malformed = root / "002_Bad_tr_2_1st_3.csv"
        for contents, error in (
            ("feature\n1\n2\n3\n", "final position"),
            ("Label,feature\n0,1\n0,2\n1,3\n", "final position"),
            ("feature,Label,Label\n1,0,0\n2,0,0\n3,1,1\n", "final position"),
            ("feature,Label\n1,0\n2,2\n3,1\n", "binary values"),
            ("feature,Label\n1,0\n2,\n3,1\n", "binary values"),
            ("feature,Label\n1,0\nNaN,0\n3,1\n", "NaN or infinity"),
            ("feature,Label\n1,0\ninf,0\n3,1\n", "NaN or infinity"),
            ("feature,Label\n1,0\n\n3,1\n", "NaN or infinity"),
            ("feature,Label\n1,0\ntext,0\n3,1\n", "numeric value"),
            ("feature,Label\n0,1,0\n1,2,0\n2,3,1\n", "numeric value"),
            ("Label\n0\n0\n1\n", "feature columns"),
            ("feature,feature,Label\n1,2,0\n2,3,0\n3,4,1\n", "unique names"),
        ):
            malformed.write_text(contents)
            rejected(lambda: read_tsb_series(malformed), error)
    print("verify_tsb_ad_data: OK (CSV alignment, declared split, normalization, labels and failures)")


if __name__ == "__main__":
    main()
