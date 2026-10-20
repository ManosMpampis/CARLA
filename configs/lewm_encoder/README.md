# Shared LEWM encoder experiments

Both phase-two frameworks use the same phase-one LEWM training. Configs and
results are grouped by that encoder experiment:

```text
configs/lewm_encoder/
  time/
    phase1.yml
    reconstruction/default.yml
    cross_attention/observed_input.yml
    cross_attention/context_input.yml
  frequency_aux/
    phase1.yml
    reconstruction/default.yml
    reconstruction/aux_crop.yml
    cross_attention/observed_input.yml
  _templates/                 # shared phase-two defaults
```

Six encoder presets are available: `time`, `frequency`, `time_aux`,
`frequency_aux`, `time_aux_no_film`, and `frequency_aux_no_film`. Each has
default reconstruction and all six cross-attention arms. The four presets
with an auxiliary head also have `reconstruction/aux_crop.yml`.

```text
results/<dataset>/lewm_encoder/<phase1 experiment>/
  phase1/[<version>/]<machine>/jepa_<phase1 tag>/
  reconstruction/<phase2 experiment>/[<version>/]<machine>/jepa/
  cross_attention/<phase2 experiment>/[<version>/]<machine>/jepa_<phase2 tag>/
```

Checkpoints, TensorBoard logs, calibration, scores, and metrics live in each
final `jepa*` directory. AE/VAE keep their framework/experiment layout.

AE, VAE, LEWM pretraining/adaptation, reconstruction, and cross-attention
validate after every training epoch. All use top-level `eval_every` for test monitoring: the default is
10, evaluation runs at each multiple and at the final epoch, and `0` disables
it. Set `common.eval_every` in a suite manifest to apply the same cadence to
every arm. The older `stage_c.eval_every` is a fallback when top-level
`eval_every` is absent. At every evaluation, all three procedures (window,
time-series, non-overlap window) print calibrated/oracle F1, event F1,
thresholds, and VUS PR/ROC, and append full metrics with the epoch to
`test_evaluations.jsonl`. TensorBoard receives the same metrics under `test/`.

Validation and periodic `test/*` TensorBoard scalars use completed epochs
starting at 1 and continue after resume. Validation improvements do not trigger
extra test evaluations; an AE/VAE validation checkpoint saved between evaluation
epochs has `evaluation: null`. Use the scoring entry point to evaluate its saved
weights. Training objectives still differ by arm, so their loss values need not
match.

Final suite scoring adds `non_overlap_window` alongside `window` and `timeseries`.
`eval_window_size` sets the maximum VUS buffer parameter in original timesteps
(default: 100), with no hidden multiplier. Time-series VUS sweeps 0 through this
value. Window-level evaluation converts it to score samples using
`max(1, eval_window_size // stride)`; non-overlap evaluation uses the scored
region length as its stride. This conversion is approximate at coarse resolution.
The setting also applies to point-adjusted VUS. Changing this convention requires
re-scoring existing runs to compare their VUS values.
For every reconstruction experiment, suite scoring forces `score_aux_crop: false`
to match training-time test monitoring, even when the experiment config enables
auxiliary crops. The effective value is recorded in `evaluation_settings`.
It reuses the scorer with evaluation stride equal to the scored region length
(the full input window, or the target crop for cross-attention) and
`include_last_window: false`. Each covered timestep contributes to exactly one
window score; window labels still use any anomalous timestep in that region.
Input context can overlap, and initial context or an incomplete tail can remain
unscored. `coverage.non_overlap_window` records the actual coverage and stride.
This additional pass has its own clean-tail threshold; its metrics are under
`test/non_overlap_window/*` in `suite_evaluation` and
`evaluation.non_overlap_window` in `metrics.json`. Scores, intervals, labels,
predictions, and coverage counts use the `non_overlap_window_` prefix in
`scores.npz`. Training-time monitoring retains its original two families.

Final suite scoring reports `test/<procedure>/oracle/calibration_quantile`
in `suite_evaluation` and the corresponding `evaluation.<procedure>.oracle`
fields in `metrics.json`. This inverts the default linear calibration quantile
on the held-out clean train tail at the test-oracle F1 threshold.
`clean_anomaly_fraction` is the fraction of those clean scores strictly above
the oracle threshold. `calibration_threshold` shows the threshold produced by
the reported quantile; `calibration_threshold_in_range` is false if the oracle
lies outside the clean-score range, in which case the quantile is clamped to
0 or 1. Window and timeseries quantiles can differ. To reproduce a reported
quantile in suite scoring, set `calibration_kwargs.quantile` (for example under
the manifest's `common`). These are test-oracle diagnostics, not a normal-only
tuning procedure, and do not change the configured calibration.

TensorBoard keeps `threshold` and `calibration_quantile` for each threshold
source. The additional `calibration_threshold`,
`calibration_threshold_in_range`, and `clean_anomaly_fraction` diagnostics
are retained in JSON reports only. Existing event files retain their old tags.

AE/VAE training and final scoring export `decision_trace.html` and
`decision_trace.json` alongside the weights. The HTML replays test-window
arrivals using validation-selected weights and a clean-tail threshold; labels
only annotate the display. Purple bands show ground-truth anomalies, colored
dots show predicted decisions, and two white lines mark the current checking
interval `[start, end)` rather than the input/context window.

Use the arrival slider to replay updates and the timestep slider to choose a
position inside that arrival's checking window. The top chart shows the latest
running mean at each scored timestep. The lower chart tracks the selected
timestep's mean only across arrivals that scored it, up to the replay position.
Scroll to zoom, drag to pan, or use the chart buttons; “Show checking window”
fits the top chart to the current checking interval. The default replay contains
the first 256 windows; set `decision_trace.max_windows: null` for all windows or
`decision_trace: false` to disable export.

## Naming and configuration

`phase1_experiment` names the shared encoder. `framework` selects
`lewm_encoder`, `reconstruction`, or `cross_attention`. `experiment_name`
names the phase-two variant within that encoder and framework. `--version`
optionally names a run and can be reused to resume. Without `--version`,
the run folder is omitted: names and tags select a stable path, with no generated
date. Reusing that path resumes training; changing a name or tag selects a new
path. Explicit older dated versions remain accessible.

Each child config uses relative `extends` paths. Parents merge left to right,
then the child overrides them. Nested mappings merge recursively; changing
the loss criterion replaces its argument mapping. Encoder/backbone/auxiliary
settings therefore come from one `phase1.yml`, while phase-two training and
loss settings come from `_templates`.

For a smaller cross-attention variant, create
`time/cross_attention/small_input.yml`:

```yaml
extends: observed_input.yml
experiment_name: small_input
cross_attention_kwargs:
  qk_channels: 4
  value_channels: 4
optimizer_kwargs:
  lr: 0.001
```

This writes under `lewm_encoder/time/cross_attention/small_input/`. Modify a
child to change one experiment; modify a template to change all its children.
Use `load_experiment_config` from `utils.config` when reading inherited YAML
in Python; `yaml.safe_load` alone returns only the child fields.

The default attention crops are `null`, meaning the two halves of the feature
map at any encoder stride. Explicit `context_crop` and `target_crop` remain
configurable intervals in feature-token coordinates.

## One encoder, both phase-two frameworks

Train phase one once through either framework entry, then train its children:

```bash
./venv/bin/python lewm_reconstruction.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/time/phase1.yml \
  --fname machine-1-1.txt --version smd_v1

./venv/bin/python lewm_reconstruction.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/time/reconstruction/default.yml \
  --fname machine-1-1.txt --version smd_v1

./venv/bin/python lewm_cross_attention.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/time/cross_attention/observed_input.yml \
  --fname machine-1-1.txt --version smd_v1
```

Phase-two initialization finds that encoder checkpoint automatically. To use
an encoder from another run, set `phase1_version` in YAML or pass
`--phase1-version`. An explicit `pretrained_from` overrides automatic lookup.
Resume uses the phase-two checkpoint and does not require the original
phase-one file. Score with the same child config and `--score`.

Every model entry supports `--score`: `lewm.py`, `lewm_reconstruction.py`,
`lewm_cross_attention.py`, `carla_ae.py`, and `carla_vae.py`. Use the same config,
machine, and version as training; scoring writes reports alongside the saved
weights without training another epoch. The two-phase entries also score
phase-one configs using their original LEWM model. Reconstruction has no
separate scoring config, including the auxiliary-crop arm.

```bash
./venv/bin/python lewm_reconstruction.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/time/reconstruction/default.yml \
  --fname machine-1-1.txt --version smd_v1 --score
```

AE/VAE use `best_validation_loss.pth.tar`; LEWM and its phase-two frameworks
use their best-validation `model.pth.tar` (LEWM can fall back to its resume
checkpoint). `--score_checkpoint` selects an explicit weights file.

The PSM manifest uses `pretrained_experiment: lewm_encoder/<phase1 name>` to
train each source once before its dependent arms. Complete phase-two keys
include the parent, e.g. `lewm_encoder/time/cross_attention/observed_input`.
The same phase-two name under another encoder is a separate experiment and
summary row. See `configs/psm/README.md` for suite commands and adding entries.

A manifest phase-one variant can reuse `time/phase1.yml` with a new name such
as `time_lr_0.1`; the suite gives it its own encoder parent. Point a child at
`pretrained_experiment: lewm_encoder/time_lr_0.1` to log that child beneath the
new parent automatically.

Existing results are preserved in their original directories. Use explicit
`pretrained_from`/`score_checkpoint` paths to read older checkpoints; new runs
use this hierarchy. Legacy configs without the grouping fields keep their
original output paths.

Details: [phase-one arms](PHASE1.md), [cross attention](CROSS_ATTENTION.md).
