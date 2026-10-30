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
share the same logging and snapshot policy. Training and validation log every
loss component once per epoch, including `reconstruction loss` and `latent_var`.
Variance is the sample-weighted mean of per-batch, per-channel population
variance across samples and latent positions (averaged across latent levels).
For phase-one and latent-target heads, `reconstruction loss` denotes their
latent prediction objective. For AE/VAE it is MSE; the mirrored head uses L1.
The total `loss` includes the arm's regularizers and selects validation weights.

`eval_every` defaults to 10 and evaluates at multiples and the final epoch;
0 disables test monitoring. `stage_c.eval_every` is a fallback if the top-level
setting is absent. Three independent TensorBoard sections and CSV histories are:

- `Test_window` / `Test_window.csv`: overlapping windows, one mean score per window.
- `Test_non_overlap` / `Test_non_overlap.csv`: disjoint scored regions.
- `Test_timeseries` / `Test_timeseries.csv`: overlap-averaged timestep scores.

The reported metrics are event F1, F1, range F1, affiliation F1, PA F1,
VUS PR/ROC and AUC PR/ROC. Calibrated and oracle decisions stay separate;
continuous-score metrics appear once per mode. Ordinary decision metrics use
`f1` and `event_f1`; adjusted F1 is `pa_f1`. TensorBoard also records
`<source>/threshold` and `<source>/quantile` for calibrated and oracle decisions. PA F1 runs last on a private
prediction array. Event intervals use exclusive end indices; the PA wrapper
also handles events starting at index zero. Undefined score metrics (single-class labels) are null in JSON
and empty in CSV. AUC PR uses trapezoidal PR area; range F1 uses flat positional
bias and alpha=0.2 existence reward. Epochs are one-based in logs and snapshots.
`test_evaluations.jsonl` retains the complete scheduled evaluation history;
`train.csv` and `validation.csv` retain the individual epoch losses and variance.

All runners save:

- `best_validation_loss.pth.tar` for the best validation loss.
- `best_test_<mode>_oracle_f1.pth.tar` for each mode's best oracle F1.
- One rolling `checkpoint.pth.tar`, overwritten at `save_every` multiples
  (default 10), evaluation epochs, and the final epoch. No `epoch_NNNN` archives
  are created. `save_every: 0` disables periodic saves.
- `final.pth.tar` at completion.

Each checkpoint is saved directly under its stable filename as a regular file.
`model.pth.tar` stores the best validation snapshot; `model_best_eval.pth.tar`
stores the best timeseries oracle snapshot. `best_metrics.json` records the latest
epoch and a `checkpoints` entry with the epoch and selection for each saved file.
Existing epoch archives from older runs are preserved.

Snapshots include weights, optimizer/scheduler state, configuration, current
losses/evaluation, selection metadata and best-metric tracking for resume.
Resume discards stale scalar/CSV history beyond the saved epoch while keeping
best-snapshot selection values, including improvements after a periodic save.
Validation selection stays independent of test-oracle selections. Between
evaluation epochs a snapshot has `evaluation: null`.

Final `--score` and evaluation-suite passes create no TensorBoard writer.
They export `metrics.json`, `calibration.json`, and, per mode,
`Test_<mode>_metrics.csv`, `Test_<mode>_scores.csv`, and `Test_<mode>.html` using
the three names above. These replace the old `scores.npz` and
`suite_evaluation` event stream. Suite summaries also have three independent
`Test_<mode>_summary.csv` files. Existing artifacts are left in place.

HTML figures reuse the original interactive interface: logarithmic scale by
default, a linear-scale option, scroll/button zoom, drag pan, reset controls,
timestep selection, tooltips, and a full anomaly overview. The primary slider
moves through the actual evaluation checking windows and shows their start,
end and width, separately from the input/context window. Both charts highlight
the selected checking region. All original timestep coordinates and all exact
scores are retained once in compact typed arrays. The overview caps visible SVG
marks per pixel; zooming reveals individual timesteps. Wheel/drag redraws are batched. There is no resize-observer feedback loop.
Metrics and score CSVs use full resolution. Detailed window-arrival replay
remains optional: `decision_trace: {enabled: true, max_windows: 32}`.

Non-overlap evaluation uses the scored region length as its stride and never
adds an overlapping tail window. Initial context and incomplete tails can
remain unscored; coverage is recorded in JSON. Reconstruction evaluation forces
`score_aux_crop: false` to match training-time monitoring. Cross-attention
`context_crop` and `target_crop` are expressed in input timesteps.

`eval_window_size` is converted to window-score units for window evaluation.
Oracle calibration diagnostics (equivalent clean-tail quantile and anomaly
fraction) remain in JSON; they do not change the configured clean calibration.

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
use their best-validation `model.pth.tar` . `--score_checkpoint` selects an explicit weights file.

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
