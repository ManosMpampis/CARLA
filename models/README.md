# Model assembly and reconstruction heads

## One training command

Select `framework: ae`, `framework: vae`, or `framework: lewm` in a
[training config](../configs/training). LeWM's `engine`, `backbone`,
`encoder`, `action`, `predictor`, `head`, and `scorer` are sibling
settings alongside `framework`. The `data`, `loss`, and
`train_hyperparameters` sections configure the rest of the run.

```bash
./venv/bin/python train.py --config_exp configs/training/lewm.yml
./venv/bin/python train.py --config_exp configs/training/guided_forecast.yml
./venv/bin/python train.py --config_exp configs/training/guided_forecast.yml --score
```

The first command trains masked reconstruction. The second initializes
the forecast backbone from the first run's encoder and trains a fresh
encoder and predictor. The backbone is frozen by default. Set
`backbone.pretrained_from` to the source checkpoint when using a different
results directory or named version. A complete run checkpoint takes
precedence when resuming.

Set `aggregate_output_strategy: all` to evaluate window, non-overlap
window, and timeseries scores. Choose `window`, `non_overlap`, or
`timeseries` individually, or list the ones you need. The selection
applies to both periodic test evaluation and `--score`.

## Model assembly and earlier experiments

Module registries and experiment assembly live in `models/__init__.py`.
The composable `LeWM` in `models/lewm.py` applies the engine, optional
backbone, encoder, action, predictor, and optional raw-output head. The
same model class runs both new presets with `lewm.py`.

- `configs/lewm_encoder/composable/masked_frequency.yml` injects synthetic
  anomalies during training and validation, supervises the mask logits,
  conditions the STFT predictor with FiLM, and trains a raw reconstruction
  head. Scoring uses the auxiliary's proposed mask on the observed input.
- `configs/lewm_encoder/composable/guided_forecast.yml` encodes history and
  future crops separately. History supplies queries and values, and the
  observed future crop supplies keys through a TCN. The score map covers
  only the future crop and is available once those observations arrive.

A `lewm_modules.backbone` can be placed before the encoder. It is frozen
by default. Use `pretrained_from` with `checkpoint_prefix` (default
`encoder.`) to load its weights when starting a new run. To add a raw head
after latent pretraining, use `stage: adapt`, `stage_b.mode: head_only`, a
positive `w_recon` and the source checkpoint in `pretrained_from`. Crop coordinates
in the composable forecast engine are in raw timesteps.

A discrete action module embeds integer action IDs supplied to
`model(..., action=ids)`. With equally sized current and target crops, it
can learn action-conditioned transitions with either FiLM predictor.
`models.action_planning.plan_discrete_actions` searches candidate action
sequences in eval mode against an explicit goal window.

The historical arms retain their own configuration and scoring semantics.

| Framework | Builder | Training/scoring entry |
| --- | --- | --- |
| LEWM phase one | `get_lewm_model` | `lewm.py` |
| LEWM reconstruction | `get_recon_model` | `lewm_reconstruction.py` |
| Cross-attention LEWM | `get_cross_attention_model` | `lewm_cross_attention.py` |
| Deterministic AE | `get_ae_model` | `carla_ae.py` |
| VAE | `get_vae_model` | `carla_vae.py` |

Both `lewm_cross_attention.py` and `lewm_reconstruction.py` dispatch phase-one
configs to the existing LEWM training loop. Each framework uses its own single
entry for pretraining, phase two, and scoring. Their models share
`models/recon_head.py`, as do the AE/VAE models. Original builder import names
at the root training entries remain aliases for compatibility with notebooks.

For LEWM reconstruction, `stage: pretrain` (aliases `pretext`, `phase1`) trains
the original latent predictor; `stage: recon` (aliases `phase2`, `adapt`) trains
the mirrored reconstruction head; `stage: score` evaluates the saved model.
The former `carla_recon.py` entry has been removed. The stage normally comes
from the YAML and can also be overridden with `--stage`.

All five model entries accept `--score` with the same experiment config,
machine, and version used for training. No separate scoring YAML is needed.
The two-phase entries select the phase-one or phase-two scoring model from
the experiment's criterion. `--stage score` remains supported for older callers.

```bash
./venv/bin/python lewm_reconstruction.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/frequency_aux/phase1.yml \
  --fname machine-1-1.txt --version smd_v1
./venv/bin/python lewm_reconstruction.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/frequency_aux/reconstruction/default.yml \
  --fname machine-1-1.txt --version smd_v1
./venv/bin/python lewm_reconstruction.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/frequency_aux/reconstruction/default.yml \
  --fname machine-1-1.txt --version smd_v1 --score
```

Phase two resolves the shared phase-one checkpoint automatically from
`phase1_experiment` and `phase1_version` (defaults to the current version).
An explicit `pretrained_from` still overrides it. Phase-two resume uses its own complete checkpoint,
so it no longer needs the original pretraining file.

## Automatic mirroring

`build_mirrored_head(encoder)` derives hidden channels, block count, and reverse
strides directly from the encoder, using the established reconstruction
contract: one ConvTranspose1d per encoder block plus a final 1x1 output
projection. For example:

```text
Encoder:       input C -> D1 -> D2
Reconstructor: D2 -> D1 -> D1 -> input C
```

This mirrors the encoder's channel ladder and block strides, rather than
mathematically inverting each individual convolution within its residual blocks.

Cross-attention keeps its Query/Key and Value widths configurable. The
attention output first passes through a 1x1 projection from Value width to
the encoder's final width, then enters the automatically derived reconstructor:

```text
Q/K/V -> narrow cross attention -> Value width -> encoder width -> mirrored head
```

That projection preserves the encoder's full hidden channel ladder; attention
width never replaces an encoder width. There is no configurable hidden-head
channel list to keep synchronized with the encoder.

The raw-input arm reverses encoder strides and projects to raw input channels.
Feature and Query targets live at feature-token resolution, so those arms keep
transpose strides at 1 and change only the final output channel count. All
three arms derive hidden channels and depth from the same encoder definition.

Cross-attention phase two freezes its feature extractor by default.
`cross_attention_kwargs.freeze_encoder: false` allows gradients and training
BatchNorm/dropout behavior in that extractor. The freeze setting is persisted
in the task metadata with the reconstruction architecture.

The corrected cross-attention reconstructor has different weights/shapes from
the initial implementation. Use a fresh run/version for phase-two training;
old cross-attention phase-two checkpoints are incompatible. Existing phase-one,
LEWM reconstruction, AE, and VAE checkpoint layouts are unchanged.

```bash
./venv/bin/python tests/verify_model_builders.py
./venv/bin/python tests/verify_lewm_reconstruction.py
```
