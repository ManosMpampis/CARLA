# TimeseriesAggregator

`utils/timeseries_aggregation.py` maps model anomaly-score outputs onto an original timestep axis or a transformed window axis. It serves offline checking and online inference. It does not calculate thresholds, predictions, or metrics. Existing training/evaluation callers still use the legacy scorer until explicitly migrated; existing result files are unchanged.

## Configuration

| Argument | Meaning |
| --- | --- |
| `input_window` | Number of input timesteps supplied to the model. |
| `stride` | Distance between input starts, in original timesteps. |
| `output_window` | Temporal output length after reduction and repetition. |
| `model` | Module exposing `.score(x)`, or a callable returning a `(B,L)` score tensor. Inputs are `(B,C,W)`. Module dictionaries use `fused`. |
| `time_offset=0` | Raw/processed output start relative to the input start, in original timesteps. Positive values map forward. |
| `check_mask=None` | Optional half-open `[start,end)` interval in original timesteps relative to the input start. Intersect with the mapped output region to retain only checked positions. |
| `output_fn=None` | Preserve raw temporal scores; their length must equal `output_window`. |
| `output_fn="mean"` / `"sum"` | Reduce each raw temporal output, then repeat its scalar to `output_window`. Reduction occurs before masking. |
| `data_fn=None` | Aggregate mapped output contributions per original timestep. |
| `data_fn="window"` | Transform labels to one sample per stride-grid input window. Each label is `any()` over the mapped, checked raw-output region, not the entire input. |
| `boundary_policy="zero"` | Uncovered prefix/suffix scores: `zero`, `crop`, or `error`. |
| `gap_policy="error"` | Uncovered internal positions: `error` or `zero`. |
| `device=None` | Infer from a module/bound method's parameters, otherwise CPU. Set explicitly for callable adapters around CUDA models. |

Every mapped position accumulates a sum and a contribution count. Final scores are sums divided by counts. Original labels are never zero-filled or altered; boundary cropping removes matching score and label positions. The actual counts determine coverage, including short series, missing windows, forecasts, and incomplete tails. For contiguous outputs, retained output length at least stride is the interior coverage condition; it does not guarantee boundary coverage.

In window mode, output positions refer to successive transformed window samples. A typical `output_window=1` produces one score for its input-window sample. Larger output lengths also contribute to subsequent window samples and are overlap-averaged there. The checked label interval advances by stride for each subsequent transformed sample. Its width comes from the raw model output and check mask, independently of scalar reduction.

## Full reconstruction examples

These assume `.score()` produces 100 temporal scores. Construct the checking loader with matching `dataset.wsz` and `dataset.stride`; changing aggregator geometry does not silently rebuild a loader.

```python
from utils.timeseries_aggregation import TimeseriesAggregator

# Non-overlapping inference, retaining one score per timestep.
non_overlap_timesteps = TimeseriesAggregator(
    input_window=100, stride=100, output_window=100,
    time_offset=0, check_mask=(0, 100),
    model=model, output_fn=None, data_fn=None,
)
result = non_overlap_timesteps(loader_stride_100)

# Non-overlapping windows: one score and one label per window.
non_overlap_windows = TimeseriesAggregator(
    input_window=100, stride=100, output_window=1,
    time_offset=0, check_mask=(0, 100),
    model=model, output_fn="mean", data_fn="window",
)
result = non_overlap_windows(loader_stride_100)

# Timeseries: average overlapping timestep contributions.
timeseries = TimeseriesAggregator(
    input_window=100, stride=10, output_window=100,
    time_offset=0, check_mask=(0, 100),
    model=model, output_fn=None, data_fn=None,
)
result = timeseries(loader_stride_10)

# Window: overlapping input windows become separate evaluation samples.
windows = TimeseriesAggregator(
    input_window=100, stride=10, output_window=1,
    time_offset=0, check_mask=(0, 100),
    model=model, output_fn="mean", data_fn="window",
)
result = windows(loader_stride_10)
```

The shorter convenience factory has the same settings:

```python
from utils.timeseries_aggregation import reconstruction_aggregator

aggregator = reconstruction_aggregator(
    model, input_window=100, stride=10, mode="timeseries",
)
# Other modes: "window", "non_overlap_window", "non_overlap_timeseries".
# Non-overlap helpers set stride=input_window.
```

## Checked label transformation

A model returns 100 values and the checked region retains only positions 50–99. For an input starting at `s`, the transformed label must be:

```python
window_label = int(labels[s + 50:s + 100].any())
```

Configure it as:

```python
aggregator = TimeseriesAggregator(
    input_window=100, stride=10, output_window=1,
    model=model, output_fn="mean", data_fn="window",
    time_offset=0, check_mask=(50, 100),
)
```

`mean` still reduces the raw model output before repetition and masking. `check_mask` limits the checked label region and which temporal output positions contribute; it does not redefine the reduction function.

For partial reconstruction producing 50 values for the second half of a 100-step input, use `time_offset=50`, `output_window=50`, and `check_mask=(50,100)` on the timestep axis. For a 50-step forecast after that input, use `time_offset=100`, `output_window=50`, and `check_mask=(100,150)`. Positive offsets leave an uncovered prefix; negative offsets may leave an uncovered suffix. Boundary policy controls the returned axis.

## Loader and streaming contracts

A checking loader exposes `dataset.series` and optionally binary timestep `dataset.targets`. Batches contain `ts` shaped `(B,W,C)` and `meta.start_idx` with original-series-local starts. `drop_last` must be false. Starts must follow the configured stride grid. Duplicate starts are rejected; batch reordering is allowed. Aggregate each series separately. Incomplete final input windows are not created or padded automatically.

```python
# Each full-loader call clears previous aggregation state.
result = aggregator(checking_loader)

# Stream an equivalent sequence of already windowed batches.
aggregator.reset(series_length=len(series), labels=labels)  # labels optional
snapshot = aggregator.update(batch)
snapshot = aggregator.result()
final = aggregator.result(finalize=True)
aggregator.reset()
```

Use `series_length` for complete boundary checking in streaming inference. Without it, the currently known axis ends at the greatest mapped position seen. Future outputs remain zero/uncovered in snapshots with known series length; strict coverage policies are enforced at finalization, not while awaiting later batches. Keep the full history until reset: this is an incremental aggregator, not a bounded-memory streaming buffer.

Results contain `scores`, optional `labels`, `indices`, `cover_counts`, `intervals`, `axis`, `uncovered_indices`, and `finalized`. Window intervals are the checked original timestep regions and can cross a series boundary; label slicing clips to the real label array. Output positions outside the available series are discarded. Returned arrays are independent snapshots.

Verification: `./venv/bin/python tests/verify_timeseries_aggregation.py`.
