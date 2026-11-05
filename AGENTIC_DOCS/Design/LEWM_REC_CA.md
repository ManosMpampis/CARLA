# Joint LEWM latent reconstruction and cross attention

This experiment keeps the existing LEWM phase-one encoder training unchanged.
Phase two shares one frozen encoder between two independently initialized heads.
The six encoder families have `cross_attention/rec_ca.yml` child configs.
The implementation is `models/lewm_rec_ca.py`; the criterion is
`losses/lewm_rec_ca.py`. It runs through `lewm_cross_attention.py` and the
existing `cross_attention` experiment-suite runner.

## Geometry and outputs

The default window is 384 raw timesteps: history `[0,256)` and future
`[256,384)`. `history_length` and `future_length` are raw timestep counts,
which must divide evenly by the encoder stride. At stride 8 the full latent
map has 48 tokens, history has 32, and future has 16.

- Rec encodes the clean whole window and uses a mirrored, stride-one latent
  decoder to reconstruct its encoder output. Its loss supervises all 48 tokens.
  This reuses the standalone Rec decoder construction and mean L1 criterion;
  the standalone Rec reconstructs raw input, whereas this arm deliberately
  reconstructs embeddings as requested.
- CA separately encodes history and future, isolating history receptive fields.
  History supplies convolutional K/V; observed future features supply Q through
  the **unchanged causal-query TCN implementation**. The CA mirrored head
  predicts the clean separately encoded future embeddings.
- The Rec prediction's last 16 tokens are compared to the CA prediction with
  mean L1. Neither prediction is detached: agreement trains both heads.

The Rec full-window target and CA separately encoded target can differ at crop
boundaries because the pretrained encoder has noncausal convolutions. Each
branch retains its own target; agreement couples their predictions.

## TCN check and causality

The existing `CrossAttentionTCNLeWM` replaces only `qkv_encoder.query`.
K/V remain ordinary centered Conv1d layers. The TCN is a stack of left-padded,
dilated convolutions, ReLU and dropout, followed by a 1x1 convolution. It
preserves sequence length; channels change from `encoder.output_dims` to
`qk_channels` (defaults 128 to 8). It contains no recurrence or attention.
Default dilations `[1,2,4,8]` with kernel 3 give a 31-token receptive field.

The combined arm inherits that implementation directly. The user's correction
keeps the TCN on Q, not V. History provides K/V and future provides Q (the
original request's second “key” is interpreted as “query”). Since future
observations enter Q, this is conditional future reconstruction. It does not
provide a past-only forecast, and token-causal Q extraction does not make the
noncausal pretrained encoder or mirrored decoder causal.

## Objective and scoring

`loss = w_rec * L1(Rec_full, E(full))`
`     + w_pred * CA_prediction_loss`
`     + lambda_sigreg * SIGReg(injected_QKV)`
`     + lambda_sigreg_tgt * SIGReg(clean_QKV)`
`     + w_similarity * L1(Rec_future, CA_future)`

All three prediction weights default to 1.0; SIGReg weights default to 0.1.
CA retains its clean/injected training streams and existing configurable
prediction loss. Rec retains clean full-window training. The encoder stays in
eval mode with gradients disabled for both paths. Loss components are logged
separately as `rec_loss`, `ca_pred_loss`, and `similarity_loss`.

Scoring averages Rec and CA residuals on the future span only, repeats latent
scores to input resolution, and uses the existing overlap aggregation and
train-only calibration. Individual residuals are exported as `rec` and `ca`
signals. Agreement is a training term, not an extra inference score. This
fusion is an explicit initial experiment choice. Evaluation uses the configured
history/future lengths; generic evaluation crop overrides are rejected.

## Run

```bash
# Existing phase one; omit if this encoder/version has already been trained.
./venv/bin/python lewm.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/time/phase1.yml \
  --fname machine-1-1.txt --version rec_ca_v1

./venv/bin/python lewm_cross_attention.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/time/cross_attention/rec_ca.yml \
  --fname machine-1-1.txt --version rec_ca_v1

./venv/bin/python lewm_cross_attention.py --config_env configs/env.yml \
  --config_exp configs/lewm_encoder/time/cross_attention/rec_ca.yml \
  --fname machine-1-1.txt --version rec_ca_v1 --score
```

Use `--phase1-version` to reuse an encoder from another version, or
`--pretrained_from` for an explicit checkpoint. Checkpoint task metadata rejects
changes to history/future geometry or architecture when resuming.

For a suite entry (the common suite defaults can override window length):

```yaml
- framework: cross_attention
  experiment_name: rec_ca
  runner: cross_attention
  config: ../lewm_encoder/time/cross_attention/rec_ca.yml
  pretrained_experiment: lewm_encoder/time
  overrides:
    wsz: 384
```

Verification: `./venv/bin/python tests/verify_lewm_rec_ca.py` checks shapes,
agreement gradients to both heads, frozen encoder, history isolation, query
TCN causality, all six configs, checkpoint task protection, and a synthetic
train/resume/score chain. It does not establish anomaly-detection performance.
