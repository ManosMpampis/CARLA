"""Plain assertions for streaming and batch timestep/window aggregation."""
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.timeseries_aggregation import TimeseriesAggregator, reconstruction_aggregator


class Windows(Dataset):
    def __init__(self, n=25, window=8, stride=2, labels=True):
        self.wsz, self.stride = window, stride
        self.series = np.arange(n, dtype=np.float32)[:, None]
        if labels:
            self.targets = np.zeros(n, dtype=np.int64)
            self.targets[[3, 7, n - 1]] = 1

    def __len__(self):
        return max(0, (len(self.series) - self.wsz) // self.stride + 1)

    def __getitem__(self, i):
        s = i * self.stride
        return {'ts': self.series[s:s+self.wsz], 'meta': {'start_idx': s}}


class Scores(torch.nn.Module):
    def __init__(self, length=8, relative=False):
        super().__init__()
        self.length, self.relative = length, relative

    def score(self, x):
        if self.relative:
            out = torch.arange(self.length, device=x.device).expand(len(x), -1)
        else:
            out = x[:, 0, :self.length]
        return {'fused': out}


def expect_error(fn, text):
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), (text, str(exc))
    else:
        raise AssertionError(f'Expected error containing {text}')


def main():
    data = Windows()
    loader = DataLoader(data, batch_size=3)
    model = Scores(relative=True)
    agg = TimeseriesAggregator(8, 2, 8, model)
    result = agg(loader)
    expected_sum, expected_count = np.zeros(25), np.zeros(25, dtype=int)
    for s in range(0, 18, 2):
        expected_sum[s:s+8] += np.arange(8)
        expected_count[s:s+8] += 1
    np.testing.assert_array_equal(result['cover_counts'], expected_count)
    np.testing.assert_allclose(result['scores'], np.divide(expected_sum, expected_count,
                                                        out=np.zeros(25), where=expected_count > 0))
    np.testing.assert_array_equal(result['labels'], data.targets)
    assert model.training  # eval state restored after inference
    assert result['scores'][-1] == 0 and result['uncovered_indices'].tolist() == [24]

    agg.reset(series_length=25, labels=data.targets)
    first = None
    for batch in loader:
        snapshot = agg.update(batch)
        if first is None:
            first = snapshot
    streamed = agg.result(finalize=True)
    for name in ('scores', 'labels', 'cover_counts', 'intervals'):
        np.testing.assert_array_equal(streamed[name], result[name])
    assert first['cover_counts'][10] != streamed['cover_counts'][10]
    expect_error(lambda: agg.update(next(iter(loader))), 'Duplicate')
    agg(loader)  # __call__ resets, rather than counting the old stream twice

    # Reduction happens per model output; mean/sum are then repeated.
    for reduction, scale in [('mean', 1), ('sum', 8)]:
        reduced = TimeseriesAggregator(8, 2, 8, Scores(relative=True), output_fn=reduction)(loader)
        np.testing.assert_allclose(reduced['scores'][:24], 3.5 * scale)

    # One transformed sample per *input* window, with labels from the checked
    # subset of the raw model output, including when the score becomes scalar.
    window = TimeseriesAggregator(8, 2, 1, Scores(), output_fn='mean', data_fn='window',
                                  check_mask=(4, 8))(loader)
    starts = np.arange(0, 18, 2)
    np.testing.assert_allclose(window['scores'], starts + 3.5)
    np.testing.assert_array_equal(window['labels'], [data.targets[s+4:s+8].any() for s in starts])
    np.testing.assert_array_equal(window['intervals'], np.column_stack((starts+4, starts+8)))
    assert window['labels'][0] == 1
    data.targets[:] = 0
    data.targets[3] = 1  # Positive outside the first window's checked region.
    assert TimeseriesAggregator(8, 2, 1, Scores(), output_fn='mean', data_fn='window',
                               check_mask=(4, 8))(loader)['labels'][0] == 0

    # Forecast placement: input length 8, output length 4, output offset 8.
    future = TimeseriesAggregator(8, 2, 4, Scores(4), time_offset=8,
                                  check_mask=(9, 12), boundary_policy='crop')(loader)
    assert future['indices'][0] == 9 and future['indices'][-1] == 24
    np.testing.assert_array_equal(future['labels'], data.targets[future['indices']])
    assert future['cover_counts'].min() >= 1
    # Negative offsets crop past-directed scores without Python negative slicing.
    past = TimeseriesAggregator(8, 2, 4, Scores(4), time_offset=-2)(loader)
    assert past['cover_counts'][0] == 2 and past['cover_counts'][18] == 0
    # Partial reconstruction and disjoint scored regions.
    partial_data = Windows(n=24, stride=4)
    partial = TimeseriesAggregator(8, 4, 4, Scores(4), time_offset=4)(DataLoader(partial_data, batch_size=2))
    np.testing.assert_array_equal(partial['cover_counts'], [0]*4 + [1]*20)

    gap_data = Windows(n=24, stride=8)
    gaps = TimeseriesAggregator(8, 8, 4, Scores(4))
    expect_error(lambda: gaps(DataLoader(gap_data)), 'internal')
    zeros = TimeseriesAggregator(8, 8, 4, Scores(4), gap_policy='zero')(DataLoader(gap_data))
    np.testing.assert_array_equal(zeros['scores'][4:8], np.zeros(4))
    expect_error(lambda: TimeseriesAggregator(8, 2, 8, Scores(), boundary_policy='error')(loader), 'boundary')
    expect_error(lambda: TimeseriesAggregator(8, 2, 4, Scores())(loader), 'raw length')
    expect_error(lambda: TimeseriesAggregator(8, 2, 8, Scores(), check_mask=(8, 10))(loader), 'retains no')
    expect_error(lambda: TimeseriesAggregator(8, 3, 8, Scores())(loader), 'stride')

    # Multiple outputs on the window axis average over adjacent window samples.
    multiple = TimeseriesAggregator(8, 2, 2, Scores(), output_fn='mean', data_fn='window')(loader)
    np.testing.assert_array_equal(multiple['cover_counts'], [1] + [2]*8)
    np.testing.assert_allclose(multiple['scores'], [3.5] + list(starts[1:] + 2.5))
    # Unlabelled inference, callable adapter, and convenience constructors.
    unlabeled = DataLoader(Windows(n=24, stride=8, labels=False), batch_size=2)
    direct = reconstruction_aggregator(lambda x: x[:, 0], input_window=8, stride=8,
                                      mode='non_overlap_timeseries')(unlabeled)
    assert direct['labels'] is None
    np.testing.assert_allclose(direct['scores'], np.arange(24))
    for mode in ('window', 'non_overlap_window', 'timeseries'):
        stride = 8 if mode.startswith('non_overlap') else 2
        made = reconstruction_aggregator(Scores(), input_window=8, stride=stride, mode=mode)
        out = made(DataLoader(Windows(n=24, stride=stride), batch_size=2))
        assert out['axis'] == ('timestep' if mode == 'timeseries' else 'window')
    print('TimeseriesAggregator: mapping, crop labels, reduction, coverage, streaming and modes OK')


if __name__ == '__main__':
    main()
