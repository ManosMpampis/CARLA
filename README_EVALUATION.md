# Whole-series evaluation and live running mean

## TSB-AD benchmark

`benchmark_tsb_ad.py` runs the existing AE/VAE, LeWM, reconstruction, and
cross-attention experiment suite on extracted TSB-AD-U/M CSVs while retaining
the native logger, TensorBoard events, resumable checkpoints, and decision
traces. It trains on the filename's declared prefix, reserves its last 20% for
validation and calibration, and exports scores for the entire series. Labels
do not affect training or checkpoint selection. Automatic window adjustment
handles short prefixes; the score archive records coverage and interpolation
of uncovered positions.

```bash
./venv/bin/python benchmark_tsb_ad.py \
  --dataset-root /absolute/path/to/TSB-AD-M \
  --files 001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv \
  --experiments ae/default --epochs 1 --version tsb-smoke --dry-run
```

Remove `--dry-run` to train. Official metrics are deferred by default and can
run later through `evaluate_tsb_ad.py` in a separate Python 3.11/3.12 TSB-AD
environment; point `--run-dir` at the benchmark's
`<output-root>/benchmark_runs/tsb-smoke` summary directory to evaluate only
that named version (`benchmark_runs/default` for an unnamed run).
See [the TSB-AD guide](configs/tsb_ad/README.md) for dataset lists,
complete-suite commands, output artifacts, metric evaluation, and fine-tuning
with a lower learning rate or model-only initialization. The full downloaded
benchmark has not been executed as part of this integration. Existing native
evaluation commands below remain available.

## Configured checkpoint evaluation suite

Select a dataset-specific manifest (`configs/psm/evaluation.yml` or
`configs/smd/evaluation.yml`), then run:

```bash
./venv/bin/python evaluate_suite.py --manifest configs/psm/evaluation.yml --dry-run
./venv/bin/python evaluate_suite.py --manifest configs/psm/evaluation.yml
```

The evaluation manifest selects exact experiment keys from `experiments_manifest`
(the existing training suite manifest). Each experiment accepts multiple named
`checkpoints` and multiple named `evaluations`; every checkpoint runs every case.
No training or dependency training runs. All five runners are supported:
`ae`, `vae`, `lewm`, `recon`, and `cross_attention`.

```yaml
name: checkpoint_comparison
experiments_manifest: experiments.yml
common:
  device: cuda
  score_batch_size: 256
  evaluation: {input_window: 1024, stride: 10}
experiments:
  - experiment: lewm_encoder/time/cross_attention/observed_input
    checkpoints:
      - name: selected        # omitted path uses the standard selected checkpoint
      - name: epoch_100
        path: ../../results/path/to/epoch_100.pth.tar
        # config: ../../results/path/to/resolved_config.yml
    evaluations:
      - name: default_halves
        input_window: 1024
        context_crop: null
        target_crop: null
      - name: longer_context
        input_window: 2048
        stride: 16
        context_crop: [0, 192]
        target_crop: [192, 256]
```

Paths to manifests, checkpoints, explicit checkpoint configs, and optional
`output_dir` are relative to the evaluation manifest. An omitted checkpoint path
selects the standard validation-selected file; optional top-level `version`
selects a named training run. For explicit checkpoints the runner automatically
uses `resolved_config.yml` beside the weights, when present, to reproduce their
architecture and training task. Otherwise it uses the training manifest entry.
Supply `config` explicitly when the checkpoint's matching config lives elsewhere.
Raw state dictionaries and Trainer checkpoints containing `model` are supported.

Settings merge in this order: checkpoint's saved config (or training entry),
evaluation manifest `common`, experiment `overrides`, checkpoint `overrides`,
then the evaluation case's fields inside `evaluation`. Use overrides for runtime
options such as device, batch size, calibration quantile, or decision traces;
architecture and checkpoint task settings must still match the saved weights.

`input_window` is the **whole raw input length**. Cross-attention `context_crop`
and `target_crop` are half-open **feature token** intervals, consistent with
training configs: multiply by the encoder's total stride for raw timestep
boundaries. With stride 8, `[0, 192]` covers raw `[0, 1536)` and `[192, 256]`
covers `[1536, 2048)`. Set both crops to null for equal halves, or omit both to
retain the checkpoint's training crops. Explicit crops must be ordered,
non-overlapping, and inside the window; input lengths must be divisible by total
stride. Default halves require an even feature token count. Only evaluation
geometry changes after strict weight loading; query source, extraction mode,
and target domain retain their checkpoint semantics. Scores automatically map
to the target crop, so `check_mask` is unnecessary and `latency_offset` must be 0.

Every job runs clean held-out training-tail calibration, timeseries and window
evaluation, non-overlapping window evaluation, calibrated and oracle reports,
VUS, and the existing honest/point-adjust metrics. Oracle results are reported
separately; they do not choose the clean-data threshold. Decision traces follow
the usual `decision_trace` setting.

The generic evaluator requires `--manifest` and takes the dataset and series
from the referenced training manifest. Outputs default to
`results/<dataset>/evaluations/<name>/<experiment key>/<checkpoint name>/<case name>/`
for a single-series dataset. Machine datasets insert `<fname>/` after `<name>/`,
with `resolved_config.yml`, `metrics.json`, `scores.npz`, `calibration.json`,
TensorBoard events, and optional decision traces. Training artifacts are kept
intact. Each completed or failed job updates suite-level `summary.json` and
`summary.csv`; subset reruns retain other summary rows. Reusing the same name,
checkpoint label, and case label replaces that job's evaluation artifacts.
Use a new suite `name` to retain a separate comparison.

Training paths also omit repeated single-series dataset identifiers:
`results/psm/ae/default/jepa_ae/`. SMD retains its machine component, such as
`results/smd/ae/default/machine-1-1.txt/jepa_ae/`. Existing artifacts at older
paths are not moved; explicit checkpoint `path` entries can still evaluate them.

Additional CLI options: `--experiments <exact keys...>`, `--device cpu`,
`--cpu-threads 1`, and `--fail-fast`. Failed jobs are recorded and later jobs
continue unless `--fail-fast` is set; the command exits nonzero on any failure.

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
The HTML also shows a ground-truth timeline across the full test series,
with the recorded score range outlined in blue. Target anomalies beyond the
replayed prefix remain visible there even when the score chart covers only
normal timesteps (as happens for the default SMD machine).
Both score charts default to a symmetric logarithmic vertical scale to keep
small scores visible alongside large outliers, including zero or negative
channel margins. Use **Score scale** to switch to linear spacing. Axis labels,
tooltips, thresholds, and decisions retain their original score values.

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
