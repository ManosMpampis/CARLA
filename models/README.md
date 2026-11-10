# Conditional LeWM and reconstruction baselines

Use `train.py` with `configs/training/ae.yml`, `vae.yml`, `lewm.yml`, or
`guided_forecast.yml`; add `--score` for saved-weight evaluation. Model, data,
training, objective and scoring belong to one YAML. Historical classes remain
importable; deleted historical training entries are not current run commands.
See the [runtime guide](../AGENTIC_DOCS/implementation/lewm-conditional/runtime.md)
for dataset selections, concurrent jobs and checkpoint policies.

```bash
./venv/bin/python train.py --config_exp configs/training/lewm.yml --dataset psm
./venv/bin/python train.py --config_exp configs/training/guided_forecast.yml --dataset smd --fname machine-1-1.txt
```

## Masked conditional reconstruction

Corrupted observations supply the current latent, and the uncorrupted same
window supplies the reference latent through the shared encoder. A raw
auxiliary predicts soft mask logits and FiLM features. Its soft mask drives
latent token replacement in training, validation and scoring. Known injected
masks supervise auxiliary BCE only. One STFT, a learned multiscale convolutional
pyramid, and iSTFT predict reference latents. Real/imaginary channels use an
exact grouped packing inverse. Scoring injects no fresh corruption.

## Guided future comparison

Future latents pass through a TCN action encoder to become Q. Historical latents
supply paired K/V at their original resolution. Attention followed by
ConvTranspose1d(kernel=3,stride=1,padding=1) maps value width to target latent
width while preserving future token count. Historical K/V may differ in length
from future Q. Future observations must have arrived before scoring; this is
delayed conditional comparison.

The default engine consumes the entire input with approximately 2:1
history/future spans: `clamp(floor(2*N/3),1,N-1)` history feature tokens.
Absolute `context_crop` and `target_crop` optionally specify raw-timestep
intervals. Positions share an origin at history's end: historical keys are
negative and future queries start at the forecast gap.

## Objectives, shapes and scores

Both LeWM presets default to latent-only KL prediction, independent L1 scoring,
no raw decoder, shared trainable encoder/reference/action branches, and dual
SIGReg weights 0.1. Auxiliary BCE weight is 1 where applicable. There is no
stop-gradient or EMA teacher. Latent width is independent of raw sensor count.

Choose `loss.loss2_kind` independently from `scorer.mode`: `kl`, `l1`, `mse`,
or `l2`. L2 means RMS across channels per token, averaged across batch/time for
training, with finite exact-zero subgradients. KL is symmetric channel-softmax
divergence; `loss.tau` and `scorer.tau` independently default to 1. Its shared
channel-offset invariance differs intentionally from L1 scoring. AE keeps MSE;
VAE adds beta-weighted posterior KL and scores deterministic posterior means.

Main presets use stride 1 and `latent_output: unrestricted`, removing activation
on both final residual branches and their sum. `latent_output: legacy` retains
old branch semantics. One checkpoint accepts varying temporal lengths with a
fixed sensor count/order. Strided inputs/crops must align to effective feature
stride. Training requires exact latent shape agreement. Scoring reduces scalar
errors on the latent grid before linear interpolation to the raw target span.
History outside the forecast target is excluded from score coverage.

Opt in to raw reconstruction with `head: {name: mirrored}`, positive
`loss.w_recon`, and optionally `scorer.domain: raw`. The decoder mirrors the
complete backbone-plus-encoder channel ladder and stride.

## Exact encoder reuse

For AE or LeWM features as a frozen forecast backbone, extend the forecast
preset with a source and a fresh downstream encoder:

```yaml
backbone:
  name: pretrained
  pretrained_from: /path/to/best_validation_loss.pth.tar
  freeze: true
encoder:
  name: resnet
  enc_channels: [64]
  enc_strides: [1]
  latent_output: unrestricted
```

Metadata reconstructs the exact source activation, normalization, widths and
strides. Explicit architecture options must agree. Use best-training weights
when validation was unavailable. Frozen modules stay eval and retain parameters
and normalization buffers while downstream modules receive gradients.

A simple AE can train its decoder on a LeWM backbone with identity encoder:

```yaml
framework: ae
backbone: {name: pretrained, pretrained_from: /path/to/source.pth.tar}
encoder: {name: identity}
```

Alternatively place the pretrained specification in `encoder` and omit
`backbone`. AE encoders train by default; set `freeze_encoder: true` or
`encoder.freeze: true` to freeze them. Identity retains dimensional metadata so
decoder geometry includes the backbone width and stride.

Runtime calls `utils.model_contract.prepare_feature_source_config(config,
feature_schema,resume_checkpoint=None)` before the registry builder, and
`initialize_feature_source(model,config,feature_schema)` only on fresh runs
before optimizer creation. Prepared configs embed source architecture,
provenance and fingerprints. Destination resume needs no original source file.
Versioned extra-state validates task roles, crop policy, schema, architecture,
freezing, objectives and score choices rather than tensor shape alone.

Legacy source checkpoints with saved constructor config are supported through
direct feature-weight extraction; legacy predictors need not be rebuilt. Legacy
feature activations/normalization/strides are preserved. Without saved sensor
names, source validation requires matching saved collection/series and input
width, with order from the destination reader; this basis is recorded in
provenance. Source and destination normalization are both recorded, with an
explicit `normalization_changed` flag. The destination retains its own
optimization-training-only scaler; feature-function equality is checked on a
common normalized tensor. Old destination checkpoints without task metadata
need explicit semantic migration.

Builders and registries live in `models/__init__.py`.

```bash
./venv/bin/python tests/verify_conditional_models.py
```
