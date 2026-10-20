# VUS evaluation window: source review

Reviewed 2026-10-20. The configuration change described below is now implemented.

## Definition and upstream conventions

The [2022 paper, Sections 4.1–4.2](https://www.vldb.org/pvldb/vol15/p2774-paparrizos.pdf)
extends anomaly labels with graded buffers to tolerate boundary uncertainty and
score misalignment. VUS aggregates over thresholds and buffer lengths from zero
to a selected upper bound. It does not choose one decision threshold or require
that the bound equal a detector's input window.

The [2025 extension, Section 4.2 and Algorithm 1](https://arxiv.org/html/2502.13318v1)
also explicitly takes a maximum buffer length. Averaging over buffer lengths
does not remove the need to specify the finite upper bound.

The [official repository README](https://github.com/thedatumorg/VUS#slidingwindow-parameter)
suggests a representative labeled anomaly duration (its example uses the median)
or an estimated period from correlation. The period is a property of the series,
so comparing detectors need not change the metric's bound with their input windows.

There are two upstream calling conventions:

- The [public API](https://github.com/thedatumorg/VUS/blob/main/vus/metrics.py)
  passes `slidingWindow` directly to `generate_curve`.
- The [robustness experiments](https://github.com/thedatumorg/VUS/blob/main/vus/analysis/robustness_eval.py)
  pass `2 * slidingWindow` to `generate_curve`.

The [underlying implementation](https://github.com/thedatumorg/VUS/blob/main/vus/utils/metrics.py)
iterates integer buffer parameters from zero through `windowSize`, inclusive.
Each buffer extends approximately `window // 2` samples on either side of an
anomaly, with graded weights and boundary clipping. Thus a maximum parameter
of 100 permits about 50 samples per side; 200 permits about 100 per side.

## CARLA mapping before the configuration fix

`utils/reconstruction_baselines.py` converts `eval_window_size` from original
timesteps to the score sequence's index units. `metrics/vus/metrics.py` then
doubles it, matching the upstream robustness-experiment convention rather than
the public API convention.

For `eval_window_size=100`, input length 1024, and stride 10:

| Protocol | Wrapper argument | Maximum buffer parameter | Score-index unit |
| --- | ---: | ---: | --- |
| timeseries | 100 | 200 | one original timestep |
| window | 10 | 20 | one window score, normally spaced 10 timesteps |
| non_overlap_window | 1 | 2 | one scored region, here 1024 timesteps |

The `max(1, size // stride)` conversion is coarse: non-overlapping window scores
cannot resolve a 100-timestep tolerance when one score represents 1024 timesteps.
These protocols also use different labels (any anomaly in a region versus
per-timestep labels), so their VUS values are not interchangeable.

## Implemented configuration policy

`eval_window_size` now directly specifies the maximum buffer parameter in original
timesteps, defaulting to 100. The wrapper no longer doubles it. Window-level
conversion remains as described above, so the example's maximum parameters are
now 100, 10, and 1. Point-adjusted metrics also honor the configured value rather
than a hardcoded 100. Suite reports record `evaluation_settings.eval_window_size`.
Existing VUS results require re-scoring for comparison under this convention.

## Suggested timescale-selection policy (not implemented)

Document the maximum buffer parameter directly, including its units and any
multiplier. Choose a characteristic timescale once per evaluated series using a
declared policy and keep it fixed across detectors. A clean-training period
estimate avoids using test labels to choose this scale. A label-derived duration
is an upstream-supported evaluation convention, but should be disclosed and
kept separate from normal-only calibration. Do not optimize the bound for each
model's test score. Use timestep VUS as the primary comparable result; explicitly
report the discretization for window-level results.
