# Two-phase convolutional cross-attention LEWM

Phase one runs the existing `lewm.py` implementation unchanged: shared
clean/injected encoder, time-domain predictor, no auxiliary head, dual SIGReg.
`time/phase1.yml` is the shared time-predictor encoder experiment. The
sweep accepts `--phase1-config` to reuse a frequency/auxiliary phase-one arm.
Existing phase-one checkpoints can also initialize phase two directly.

Phase two freezes that encoder as a feature extractor by default, including its
BatchNorm statistics and dropout. Set `freeze_encoder: false` to train it too.
Three new Conv1d layers form the trainable LEWM encoder:
Key and Value see the first crop; Query sees the second crop in the requested
`observed` arm. The predictor is scaled cross attention followed by the same
`MirroredReconHead` used by the existing reconstruction framework. A 1x1
projection maps the narrow attention output to the feature extractor's final
width; the head then mirrors that encoder's full hidden channel ladder. There
is no Query residual connection around attention. The bottleneck can be made
narrow without changing the feature extractor or its mirrored hidden widths.

Final assembly lives in `models/builders.py`. `lewm_reconstruction.py` is the training
entry for LEWM reconstruction; cross-attention uses `lewm_cross_attention.py`.
Both share the head implementation in `models/recon_head.py`.

| Reconstruction target | Config for observed Query | Config for past-only Query |
| --- | --- | --- |
| Second crop's frozen encoder features | `time/cross_attention/observed_features.yml` | `time/cross_attention/context_features.yml` |
| Second crop's raw normalized input | `time/cross_attention/observed_input.yml` | `time/cross_attention/context_input.yml` |
| Query convolution of the clean second crop | `time/cross_attention/observed_query.yml` | `time/cross_attention/context_query.yml` |

## Reconstruction versus forecasting

When Query comes from the second window, the model already observes that
window before reconstructing it. It learns conditional reconstruction: how to
reconstruct the observed second window using values retrieved from the first.
Residuals can measure disagreement between adjacent windows, but this training
does not make a forecast of an unseen future window. The small bottleneck
restricts capacity; it does not guarantee useful anomaly separation.

The `context` configs instead obtain Query by applying its convolution to the
first crop's features, interpolated to the target token length. Training still
supervises the actual second crop. These arms predict from the past alone.
The raw-input arm is the one that directly forecasts telemetry; features and
Query arms forecast their respective representations.

For forecasting, `feature_extraction: separate` encodes the two raw crop
intervals independently. This prevents the noncausal convolutional feature
extractor from mixing second-window values into first-window features. `full`
extracts one full-window feature map and then slices it, exactly as requested;
the observed configs use this mode. Full extraction lets features near crop
edges share receptive fields across the split, and is rejected for forecasting.
Set `separate` in an observed config for an experiment that also isolates crops.

At inference, `model.score(x)` compares the second crop to its reconstruction.
It needs both windows to compute the residual. A past-only model can also
forecast before the second crop arrives:

```python
# model built from a context_input config and loaded with phase-two weights
model.eval()
# context_input: (batch, input_channels, context_input_steps), normalized as training
next_input = model.predict_next(context_input)  # (batch, input_channels, target_input_steps)
```

For a stream, buffer the two configured crop intervals to score each arriving
second window. Supply the model to the existing `RunningScorer` with the full
input window and `check_mask=model.target_input_range(input_window)` to place
its residuals on the second window. The initial context has no anomaly score.

## Training and regularization

Phase two uses clean and injected views through the same Q/K/V convolutions.
Prediction targets are always clean. The loss is

```text
w_pred * L1(prediction, chosen_clean_target)
  + lambda_sigreg * mean(SIGReg(injected Q), SIGReg(injected K), SIGReg(injected V))
  + lambda_sigreg_tgt * mean(SIGReg(clean Q), SIGReg(clean K), SIGReg(clean V))
```

`loss2_kind` also accepts `mse` and symmetric channel-wise `kl`, using the
existing LEWM loss geometry. L1 is the default for all three targets, preserving
amplitude information in raw telemetry and features. No target detach is added:
the Query target receives gradients, while raw targets and frozen features
are fixed naturally. SIGReg regularizes the trainable Q/K/V representations.
The predictor output is supervised by the reconstruction loss.

`stage_b.masking.mode: subanomaly` uses the existing anomaly injector for
denoising training and validation. Set it to `none` for normal-only conditional
reconstruction/forecasting. There is no auxiliary mask head or action input in
phase two. Checkpoint selection uses validation prediction loss only; test
labels enter only the final metric calculation. Train-tail quantile thresholds
and diagnostic test-oracle metrics are reported separately.

## Configuration

- `context_crop` and `target_crop` are half-open **feature token** intervals.
  With total extractor stride `S`, `[a,b]` maps to raw input `[a*S,b*S]`.
  The context must precede the target without overlap. Unequal lengths and
  a gap between intervals are supported. Omit both crops (or set both to null)
  for an automatic half split with an even feature length.
- Defaults use a 128-step input, context `[0,64]`, target `[64,128]`, and
  extractor stride 1. To compare two complete 128-step windows, use `wsz: 256`
  and token crops `[0,128]` and `[128,256]` at stride 1.
- `qk_channels` controls Query/Key channels; `value_channels` controls Value
  and attention output channels. `num_heads` must divide both widths. Each
  projection is one convolution with configurable odd `kernel_size`.
- Reconstructor hidden widths and depth are derived automatically from
  `model_kwargs`, using one ConvTranspose1d per encoder block plus the existing
  final 1x1 projection. No `head_channels` list is required. Only the raw-input
  arm reverses extractor strides. Features/Query heads preserve feature token
  length and use their target channel count in the final projection.
- Narrow attention reduces channel computation and limits capacity. Temporal
  attention still relates every Query token to every Key token, so attention
  cost also depends on both crop lengths.
- `score_mode: l1|l2|mse` changes residual reduction independently of training
  loss. Latent residuals expand by extractor stride to input timesteps.
- Scoring automatically maps residuals to the target crop. Optional
  `evaluation.input_window` must contain both crops. A supplied
  `evaluation.check_mask` must equal the target input range and
  `evaluation.latency_offset` must be zero; scores already have that placement.

## Run the complete chain

```bash
# Original phase one once, then all three requested reconstruction arms and scoring
./venv/bin/python experiments_cross_attention.py \
  --fname machine-1-1.txt --version cross_attention_v1

# Compare reconstruction and past-only forecasting (six phase-two arms)
./venv/bin/python experiments_cross_attention.py \
  --fname machine-1-1.txt --version cross_attention_v1 --query-source both

# Forecast raw telemetry with a smaller bottleneck
./venv/bin/python experiments_cross_attention.py \
  --fname machine-1-1.txt --version forecast_v1 \
  --query-source context --arms input --qk-channels 4 --value-channels 4

# Inspect checkpoint handoff paths without creating run directories
./venv/bin/python experiments_cross_attention.py --dry-run --query-source both
```

The sweep supports `--all`, `--limit`, `--phase1-epochs`, `--phase2-epochs`,
and `--phase1-config`. `--phase1-experiment` selects a shared encoder preset;
`--phase1-version` selects its run, defaulting to `--version`. Encoder geometry is copied
from phase one's config, preserving a 64-step context and a 64-step target.
Use the standalone harness to change window/crop geometry freely.

## Run stages separately

```bash
./venv/bin/python lewm_cross_attention.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/time/phase1.yml \
  --fname machine-1-1.txt --version smd_v1

./venv/bin/python lewm_cross_attention.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/time/cross_attention/observed_input.yml \
  --fname machine-1-1.txt --version smd_v1

./venv/bin/python lewm_cross_attention.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/time/cross_attention/observed_input.yml \
  --fname machine-1-1.txt --version smd_v1 --score
```

Standalone phase-two initialization strictly loads **all** encoder weights and
buffers from either a LEWM `model.pth.tar` or Trainer `checkpoint.pth.tar`.
Predictor/auxiliary weights from phase one are excluded. Missing or incompatible
encoder weights fail. Resuming a phase-two checkpoint requires no phase-one
file. Checkpoints validate target, Query source, crop and attention/head
semantics, preventing equal-shaped arms from silently sharing a checkpoint.
Standalone encoder configs must also match the original phase-one architecture,
including strides: legacy state dictionaries do not record convolution strides.
Use a fresh `--version` when changing an arm's configuration.
The corrected automatic mirror changes cross-attention phase-two weights and
architecture metadata. Old phase-two checkpoints require a fresh training run;
existing phase-one encoder checkpoints remain usable.

Outputs follow the shared encoder hierarchy:
`results/<dataset>/lewm_encoder/<phase1>/cross_attention/<phase2>/<version>/<machine>/jepa_cross_attention_<source>_<target>/`.
They include training/resume and best-validation weights, TensorBoard,
`calibration.json`, `scores.npz`, and `metrics.json`. Thresholds use only the
held-out clean training tail. Coverage counts exclude unscored context from
time-series evaluation and calibration. Window labels refer to the target crop.
The report includes the frozen metric engine's `honest` and
`point_adjust_comparability` sections alongside window/time-series evaluation.
Saved predictions remain unadjusted.

```bash
./venv/bin/python tests/verify_lewm_cross_attention.py
```

The assertion script tests all targets, strides, unequal crops, gradients,
frozen buffers, exact attention, future isolation, strict weight handoff,
score placement, and a synthetic phase1 -> six phase2 arms -> resume -> score
chain. Outputs for this check stay under a temporary directory.
