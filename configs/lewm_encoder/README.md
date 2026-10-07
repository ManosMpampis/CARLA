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
  phase1/<version>/<machine>/jepa_<phase1 tag>/
  reconstruction/<phase2 experiment>/<version>/<machine>/jepa/
  cross_attention/<phase2 experiment>/<version>/<machine>/jepa_<phase2 tag>/
```

Checkpoints, TensorBoard logs, calibration, scores, and metrics live in each
final `jepa*` directory. AE/VAE keep their framework/experiment layout.

## Naming and configuration

`phase1_experiment` names the shared encoder. `framework` selects
`lewm_encoder`, `reconstruction`, or `cross_attention`. `experiment_name`
names the phase-two variant within that encoder and framework. `--version`
names a run and can be reused to resume.

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
