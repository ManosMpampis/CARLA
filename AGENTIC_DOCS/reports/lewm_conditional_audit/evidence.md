# Initial probe results

Audited 2026-11-09. Source basis: the existing dirty working tree, including
untracked composable configs/train.py and the already deleted legacy scripts.
The audit did not restore or modify those files.

Environment check: `./venv/bin/python` is Python 3.12.3 with Torch 2.12.0+cu132;
CUDA is available. `import train` succeeded. The documented Python 3.14 version
is stale. Model and transfer probes ran on CPU.

## Model execution

For batch 2, channels 3, window 512:

| Preset | Target latent shape | ConvTranspose1d count | Score shape | Forward/backward |
| --- | --- | --- | --- | --- |
| masked frequency reconstruction | `[2,128,64]` | 3 | `[2,512]` | Finite |
| guided forecast | `[2,64,24]` | 0 | `[2,192]` | Finite |

Both accepted additional windows 520, 528 and 768. Forecast retained its fixed
raw crop `[320,512]` and the same score length. These checks do not establish
correct learning or useful anomaly scores.

Spectral layout probe: a two-frame, three-frequency complex tensor packed and
unpacked with the current operations did not roundtrip. First input frame
`[1+10j,2+20j,3+30j]` became `[1+4j,2+5j,3+6j]`.

ResNet latent probe: removing only the final sum activation retained the two
GELU branch outputs, with an observed minimum near -0.33993. With dropout zero,
the approximately -0.34 lower bound prevents a standard-normal output.

Explicit legacy KL probe: prediction `target+10` gave loss approximately
`1.5e-14` and L1 difference 10. **The new composable presets already override
the constructor default to L1**; this is a warning about selecting KL, not a
claim that their default is KL.

## Transfer execution

Temporary probe: `/tmp/lewm_transfer_audit.py`; temporary checkpoints:
`/tmp/lewm_transfer_audit/`. These paths are ephemeral.

Tiny AE and masked/frequency LeWM sources were each updated for one optimizer
step, saved, and loaded into a new guided forecast backbone.

- Both had exact source-encoder/destination-backbone state equality.
- Frozen backbones remained eval and all weights/BatchNorm buffers were
  unchanged after a downstream optimizer step.
- Downstream encoder absolute gradient sums were nonzero for both sources.
  A saved rerun measured approximately 16.86 (AE source) and 64.69 (LeWM source);
  see [transfer_probe.json](transfer_probe.json) for its complete output. The
  probe seeds Torch but not every corruption RNG, so exact LeWM gradient
  magnitudes vary between runs.
- Prediction shape was `[4,4,6]` for batch 4, three input features, 64 raw
  steps and 40/24 history/future crops with effective stride 4.
- Matching masked-LeWM source/destination features had mean difference 0.
- AE source/destination features differed by mean 0.101288 because the destination
  used identity where the source used final GELU, despite matching weights.
- Stride-4 encoder weights loaded into a stride-2/ReLU definition without
  rejection. A changed-crop task also accepted strict state loading.

## Dataset fixtures

Temporary folders: `/tmp/lewm-dataset-audit-f_x2rfem` and
`/tmp/lewm-formats-audit-hjnna9oy`.

- Two-machine SMD fixture: current `all` yielded one corpus with 28 training
  and 8 validation windows; no window crossed a machine boundary.
- Test loading for `all` raised the explicit train-corpus-only error.
- `all-one` raised FileNotFoundError for a literal train/all-one path.
- PSM fixture with no data.file resolved filename `psm` and reused its training
  scaler at test time.
- One-feature and three-feature TSB CSV fixtures each had 160 rows, declared
  train prefix 120 and train90/validation30/test40 row splits; both reused
  the training-prefix scaler.
- Synthetic train/test normalization was inconsistent: for a short two-feature
  fixture, means were approximately `[0.5535,0.3822]` versus `[-14.8195,0.0091]`.
  The anomalous test series was used to fit its own scaler.
- Importing benchmark_tsb_ad failed because utils.tsb_ad_benchmark is deleted.

## Verification limit

Loss sample-weighting, held-out-split fallback, clean calibration and checkpoint
selection were traced in source. A complete normal-entry training, validation,
test, save/reload and resume experiment has **not** been run during this initial
audit. It is an explicit acceptance step after design agreement and corrections.
