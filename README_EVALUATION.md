# Whole-series evaluation and live running mean

The score stage cuts a series into full input windows, asks the model for a
score map for each window, places those scores on the series timeline, and
averages scores that land on the same timestep. JEPA, reconstruction, and
AE/VAE evaluation use the same placement and cover-count mean. Live inference
can update that mean incrementally as windows arrive.

A final `stage: score` run automatically writes `decision_trace.html` and
`decision_trace.json` beside `scores.npz` and `metrics.json`. The metrics
report lists both paths and how many windows were replayed. This covers
LEWM, reconstruction, cross-attention, and AE/VAE score entries. Periodic
training evaluations do not export a trace. By default, the graph replays
the first 256 input windows so the output remains manageable:

```yaml
decision_trace:
  max_windows: 512
```

Set `max_windows: null` to replay every evaluation window, or
`decision_trace: false` to disable trace output.

Add an `evaluation` block to an experiment YAML to configure this procedure
without changing training `wsz` or training `stride`:

```yaml
wsz: 100                    # training window
stride: 10                  # training stride
evaluation:
  input_window: 100         # input passed to model.score
  stride: 1                 # starts at 0, 1, 2, ...
  check_mask: [0, 100]      # output positions relative to input start
  latency_offset: 0         # signed shift on the series timeline
```

If `evaluation` is omitted, `input_window` and `stride` inherit `wsz` and
`stride`; the output is mapped across the full input window, latency is zero,
and all output positions contribute to the mean. The model must accept
`evaluation.input_window`; some architectures require it to be divisible by
their encoder stride. `eval_window_size` is a separate metric parameter for
range/VUS evaluation, not the input length.

## Placement rules

For an input window starting at series index `s`:

1. `input_window: W` supplies `series[s:s+W]` to the model.
2. `check_mask: [a, b]` places the model's output of length `b-a` on
   `series[s+a:s+b]`. The end is exclusive. The span must exactly match the
   score map length returned by `model.score`.
3. `latency_offset: d` shifts that placement to `[s+a+d, s+b+d)`. Positive
   values move scores later; negative values move them earlier.
4. Each covered timestep receives the arithmetic mean of all output values
   placed there by complete input windows.

An output can extend beyond the input window. For example, a 100-step input
with a 50-step forecast uses `check_mask: [100, 150]`. Output positions beyond
the series boundary are clipped. Windows whose entire output is
outside the series are omitted from the returned window intervals.

## Common configurations

**No overlapping input windows** (for a 100-step model):

```yaml
evaluation:
  input_window: 100
  stride: 100
  check_mask: [0, 100]
```

Starts are `0, 100, 200, ...`. A remainder shorter than 100 has no complete
input window and is unscored. No off-grid final window is inserted when
`stride >= input_window`.

**Every possible complete input window**:

```yaml
evaluation:
  input_window: 100
  stride: 1
  check_mask: [0, 100]
```

For a series of length `T`, this scores `T-99` windows. It uses more model
calls but gives each interior timestep up to 100 contributions.

**A 50-step output belonging to the second half of a 100-step input**:

```yaml
evaluation:
  input_window: 100
  stride: 1
  check_mask: [50, 100]
```

**A 50-step forecast after a 100-step input**:

```yaml
evaluation:
  input_window: 100
  stride: 1
  check_mask: [100, 150]
  latency_offset: 0
```

For an architecture whose output is delayed by three timesteps relative to
the desired label alignment, set `latency_offset: -3`.

## Live inference

`RunningScorer` scores one complete window at a time and keeps a running sum
and count for every mapped timestep. Each update returns the current means
for the timesteps changed by that window. A later window can revise a
previously returned mean. After all windows arrive, its means match
whole-series evaluation over the same windows. Supply the clean calibration
threshold to also track anomaly decisions and every decision transition.
Use the clean-data threshold from calibration, such as `threshold_fused` in
the shared JEPA score report or `timeseries_threshold` in the reconstruction
score report. The live comparison uses `score >= threshold`.

```python
import numpy as np
import torch
from utils.scoring import RunningScorer

running = RunningScorer(model, torch.device("cpu"), input_window=100,
                        check_mask=[0, 100], latency_offset=0,
                        threshold=calibrated_threshold,
                        record_history=True)
for start in range(0, len(series) - 100 + 1):
    update = running.update(np.asarray(series[start:start + 100]), start)
    # update["indices"] gives affected timesteps; update["scores"] gives
    # their current means; update["decisions"] gives current anomaly flags.
    latest_mean = running.score_at(start + 99)
    latest_decision = running.decision_at(start + 99)

running.save_decision_trace_html("decision_trace.html")
```

Only full `(input_window, channels)` inputs are accepted. Feed windows in
increasing start order. A current mean uses all model outputs available so
far; it may differ from the final whole-series mean until later windows
arrive. The class stores sums and counts instead of a timestep-by-output
matrix. For a stride other than one, advance `start` by that stride.
`decision_changes` records the first decision and each later flip with the
window and timestep that caused it. `record_history=True` retains every
intermediate mean for the slider graph; leave it off for long production
streams to avoid keeping all output updates in memory. The exported HTML
shows the score timeline at each window arrival and the full evolution of a
selected timestep, with anomaly/normal markers and the calibrated threshold.
`RunningScorer.from_config` accepts the same YAML placement options, plus
`threshold` and `record_history`.
Open the HTML written by `save_decision_trace_html`; the file
`utils/decision_trace_fragment.html` is only the unfilled graph template.

## Coverage, outputs, and metrics

Only complete input windows are scored. When `stride < input_window`, the
default is to add one final off-grid window to cover the tail. Set
`include_last_window: false` under `evaluation` if exact grid starts matter.
For `stride >= input_window`, no final off-grid window is added by default.

The saved `scores.npz` contains full-length timestep scores and
`cover_counts`. A zero count means no model output was mapped to that
timestep. These positions are excluded from calibration thresholds and
evaluation metrics; they should not be interpreted as measured normal
scores. The scorer retains its historical tail forward-fill in the saved
score array, while other uncovered positions are zero. Range metrics operate
on the sequence of covered timesteps, so a stride larger than the mapped
output length can create interior gaps and shorten their effective time axis.

`start_idxs` and `end_idxs` are the mapped output intervals used for
window-level labels and metrics. `input_start_idxs` and `input_end_idxs`
record the corresponding input windows. `window_scores` in AE/VAE outputs
are means over the in-range output positions. `cover_counts` counts
contributions to each series timestep.

Check the implementation with:

```bash
./venv/bin/python tests/verify_evaluation_mapping.py
./venv/bin/python tests/check_scorer_handoff.py
```
