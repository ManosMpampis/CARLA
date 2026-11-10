# Conditional LeWM experiments: audit and draft implementation specification

Status: **consolidated draft, awaiting final shared-understanding confirmation and L2 convention**. Audited the current working tree on
2026-11-09, including its uncommitted changes and deleted files. This document
does not describe HEAD alone. No model, trainer, dataset, or configuration changes
were made for this audit. Confirmed decisions are recorded in section 6;
remaining recommendations are proposals until the interview concludes.

## 1. Requested experiment

The project experiments with anomaly-detection architectures using simple AE and
VAE reconstruction baselines. Learned encoders, projections, predictors and
decoders should use convolutions; the explicitly requested cross-attention and
shape-derived transforms are allowed. Avoid flattening a temporal window into a
fixed-size learned vector, fixed-length positional tables, or dense window-sized
bottlenecks.

Two conditional-prediction variants are requested:

1. Latent reconstruction: an auxiliary head supplies an action/conditioning
   signal; a frequency-domain multiscale pyramid predicts the clean latent
   state of the same window. In the user's LeWM terminology, the corrupted
   view is the current state and the clean view is the future state; these are
   not chronologically separated crops. The default model has no raw
   reconstruction head or raw loss.
2. Guided future-state comparison: patterns extracted from the observed future
   supply an action/conditioning signal; cross-attention and a small transposed
   convolutional module predict a latent state to compare with that future.
   Future-derived features supply Q; paired historical features supply K/V.

Both LeWM models use **KL as the default latent-prediction loss and L1 as the
default anomaly score**, selected independently through configuration. The
latent loss supports `kl`, `l1`, `mse` and `l2`; the LeWM score selector supports
independent selection of those functions. Latent predictions must match their target latent channels and
length; neither must match the raw input's channel count. At effective stride 1
the score positions map directly to raw timesteps. Otherwise the scalar latent
score map can be interpolated to the scored raw target span. An additional raw
transpose head remains an opt-in reconstruction path; it is not part of the
default LeWM objective.

The trained reconstruction encoder and a trained AE encoder must be reusable
in the second experiment. Additionally, a trained LeWM encoder must be usable
in a simple AE through either of these configurations:

- As the AE backbone, with `nn.Identity` as its encoder.
- As the AE encoder, with a user-controlled `freeze_encoder` flag. AE encoders
  are trainable by default.

The effective feature path and decoder geometry must work in both cases;
an identity encoder must not hide the backbone's output width or total stride.

One checkpoint supports variable temporal lengths with a fixed sensor count
and ordering. Sensor counts may differ between independently constructed runs.
The frequency architecture is one STFT plus the learned multiscale pyramid;
multiple FFT/window resolutions are explicitly outside the requested scope.

No action optimization or planner is required. Existing discrete-action
planning code is outside this implementation scope.

The dataset interface must support:

| Collection | Selection | Required jobs |
| --- | --- | --- |
| PSM | No filename | One training job |
| SMD | Machine filename | One training job |
| SMD | `all` | One independent training job per machine |
| SMD | `all-one` | One model trained on all machine training series |
| TSB-AD-U | Series filename | One training job |
| TSB-AD-U | CSV manifest | One independent training job per listed series |
| TSB-AD-M | Series filename | One training job |
| TSB-AD-M | CSV manifest | One independent training job per listed series |

## 2. Framework interpretation

LeWM learns action-conditioned latent prediction with a shared, jointly trained
encoder and Gaussian regularization; planning is a separate use of that model.
Its published prediction objective is MSE and does not use stop-gradient or an
EMA teacher. Therefore omitting a planner is compatible with using its learning
recipe. [Primary paper, section 3](https://arxiv.org/html/2603.19312v3).

The requested learned conditioning, reconstruction supervision, convolutional
architecture and future-observation access make these **LeWM-derived
conditional anomaly models**. The guided residual is available after the target
observations arrive; it must not be described or evaluated as a past-only
forecast. This is an inference from the proposed information flow. It makes no
claim that the learned action represents a physical intervention or a causal
transition.

The actual composable presets default to L1 latent prediction and dual SIGReg,
through `utils/common_config.py:43`. The `LeWMLoss` constructor's legacy default
is symmetric channel-softmax KL. Do not confuse that constructor default with
the current training presets. If KL is explicitly selected, channelwise shifts
can produce nearly zero loss while L1 anomaly scores remain large; the selected
loss and score geometry must be explicit.

That paragraph describes the audited code. The agreed implementation changes
the default latent objective to KL for both LeWM presets while retaining L1
scoring. This geometry difference is intentional and must not be rejected or
silently changed. Retain the shared trainable encoder, dual SIGReg and
unrestricted latent outputs. The AE/VAE baseline objectives remain unchanged.

## 3. Current support and evidence

### 3.1 Models

| Capability | State | Evidence |
| --- | --- | --- |
| AE/VAE convolutional reconstruction | Present | `models/ae_baseline.py:16`; VAE posterior uses 1x1 convolutions at line 30 |
| Auxiliary-conditioned frequency reconstruction | Present with spectral correctness defect | `models/actions.py:76`, `models/steped_predictor.py:141`, `models/lewm.py:89` |
| Multiple learned frequency-pyramid scales | Present | Down/top-down/PAN paths in `models/steped_predictor.py:161` |
| Multiple independent STFT resolutions | Absent | One STFT call at `models/steped_predictor.py:151` |
| Future-pattern-derived conditioning | Present | `EncoderAction` applies its TCN to target latents, `models/actions.py:115` |
| Guided cross-attention | Present | `models/lewm_cross_attention.py:44` |
| Small forecast transpose module | Missing in shipped preset | `configs/training/guided_forecast.yml:28` sets `head: null`; guided predictor has Conv1d output projection only |
| Variable temporal lengths with one model | Partial | Length must divide effective stride, `models/lewm.py:74`; explicit forecast crops remain fixed |
| Variable sensor count with one trained model | Absent | First/last convolution widths are fixed; builder instead infers width separately for each dataset/job, `models/__init__.py:249` |

The reconstruction preset has the following information flow:

```text
clean window ── corruption engine ── corrupted current view ── encoder ── current latent
      │                                      │                              │
      │                                      └── raw auxiliary ── logits + FiLM
      └── shared encoder ── clean target latent                 │           │
                                               mask-token blend ──────────┘
                                                       │
                                      STFT → multiscale conv pyramid → iSTFT
                                                       │
                                               predicted latent → mirrored raw head
```

The forecast preset currently computes:

```text
history raw crop → optional backbone → encoder → interpolate to target length → Q, V
future raw crop  → same backbone/encoder → TCN action ────────────────────────→ K
attention(Q, K, V) → 1x1 convolution → predicted future latent
```

K and V have equal lengths but describe different spans. This is an executable
layout, not proof that it is the desired association of patterns with historical
values. An alternative in which future-derived queries retrieve historical
keys/values needs an explicit user decision; do not silently change tensor roles.

### 3.2 Confirmed model issues

**F1 — Incorrect spectral packing.** At `models/steped_predictor.py:158`, stacking
real and imaginary values creates `[B*S,T,2,F]`, which is reshaped directly into
`[B,2*S,T,F]`. Unpacking at line 177 assumes grouped real and imaginary channels.
Time frames and complex coordinates get mixed. A pure pack/unpack probe returned
`[1+4j, 2+5j, 3+6j]` for the first frame of an input whose first frame was
`[1+10j, 2+20j, 3+30j]`. Forward/backward shape checks do not detect this.

**F2 — Identity final activation does not produce an unrestricted latent.**
`models/blocks.py:188` and line 194 retain GELU on both branches of the last
ResNet block. `output_act: identity` only removes activation after their sum.
With dropout disabled, their sum is bounded below at approximately -0.34.
This prevents representing a full standard normal, the target of SIGReg.
Provide an explicitly unrestricted convolutional latent projection or a final
block whose branch outputs are unactivated. Version this architecture change;
do not reinterpret old checkpoints as the new model.

**F3 — Validation task and randomness need explicit policy.**
`train.py:76` supplies neither a validation collator nor a fixed corruption set.
`MaskedEngine.forward` samples corruption whenever `corrupt=True`, including
eval mode (`models/input_engines.py:29`). `LeWM.forward` additionally switches
from known-mask placement in training to auxiliary proposals in eval
(`models/actions.py:88`). Thus validation is a stochastic, corrupted-input,
predicted-mask task; `score()` uses uncorrupted observations. These paths may
be useful, but should be named, reproducible, and reported separately.

**F4 — Fixed forecast crops can ignore additional input.** With explicit crops
`[0,320]` and `[320,512]`, a longer accepted input still predicts the same
192-step target. It does not dynamically enlarge the future span. Choose fixed
absolute crops or a shape-derived crop policy and persist it in task metadata.

### 3.3 Encoder transfer

`train.py:63` invokes `Trainer.initialize_backbone` before optimizer creation on
new runs. The helper extracts `encoder.*` and strict-loads into `backbone`
(`utils/trainer.py:241`). Both source model types use this key prefix.

Synthetic one-step-trained sources confirmed:

- AE and masked LeWM encoder weights load exactly into the forecast backbone.
- Frozen backbones remain in eval mode after destination `train()`.
- Frozen parameters and BatchNorm buffers remain unchanged after a destination
  optimizer step; the downstream encoder receives gradients.
- Full-run resume takes precedence over source initialization in `train.py:65`.

However:

1. **Weights are checked; source function is not.** Same-shaped weights loaded
   into a stride-2 backbone from a stride-4 source without rejection. The helper
   ignores the saved configuration.
2. **Shipped AE and guided backbone activations differ.** AE defaults to the
   final GELU; the guided backbone specifies identity. Merely replacing
   `pretrained_from` with an AE path changes its feature function despite exact
   state-dictionary equality. A reduced probe measured mean feature difference
   0.101288; matching LeWM source had difference 0.
3. **Source resolution is hardcoded.** The guided preset points to one SMD
   machine and version. Dataset-wide runs need explicit source-to-job matching.
4. **Resume does not check task semantics.** Same-weight-shape forecast models
   with different crops strict-load successfully. The composable model lacks
   the legacy attention model's task extra-state checks.
5. Inference metadata records legacy model options rather than the complete
   composable modules, effective strides, conditioning source and crop policy
   (`utils/checkpointing.py:12`). Full config is saved but not validated.

### 3.4 Dataset dispatch, losses and evaluation

PSM without a filename and explicit SMD/TSB filenames work at the reader level.
TSB supports one or multiple feature columns, excludes `Label` from model
features, and uses the declared training prefix (`data/tsb_ad.py:34`). The
trainer infers channels independently per job.

Missing or incorrect paths:

- `utils/common_config.py:61` interprets SMD `all` as **one joint corpus**;
  `all-one` is interpreted as a literal filename and fails.
- The pooled corpus preserves machine boundaries and per-machine normalization,
  but has no `full_train_series`. Periodic testing first tries to open a single
  test file named `all` (`train.py:56`), which is explicitly rejected by
  `data/jepa_dataset.py:90`. Shared evaluation also requires `full_train_series`
  (`utils/reconstruction_baselines.py:289`).
- New `train.py` has no manifest/fan-out planner. `benchmark_tsb_ad.py:12` imports
  the deleted `utils.tsb_ad_benchmark`; legacy suite paths reference deleted
  training and summary modules. A surviving CLI file does not establish support.
- TSB collection U/M identity is not part of the new selection interface;
  both currently use internal source name `tsb_ad` and explicit roots.
- Synthetic test data fits its own scaler, including anomalies
  (`data/jepa_dataset.py:102`–115). This contaminates simulated test results.
- Real readers fit normalization on the complete training source before holding
  out validation. This uses validation statistics, although it avoids real test
  leakage. Short series silently reuse training rows as validation
  (`training_split_bounds`, `data/jepa_dataset.py:20`).
- Pooled normalization caches use filenames/existence without validating root,
  source freshness or preprocessing configuration (`data/jepa_dataset.py:268`).

Working contracts to preserve:

- Batch-mean losses are aggregated with sample-count weights during training
  and validation (`utils/trainer.py:107`, line 139). There is no optimizer
  gradient-accumulation option; that is distinct from epoch loss aggregation.
- Validation runs eval/no-grad. LeWM targets remain trainable through the shared
  encoder; target detachment is not part of these presets.
- Total LeWM loss is the sum of separately weighted prediction, auxiliary,
  raw-reconstruction and two SIGReg terms (`losses/lewm.py:90`). Disabled terms
  need not supply their operands. AE uses MSE; VAE adds beta-weighted KL.
- Calibrated thresholds use clean training score distributions; test-oracle
  snapshots are saved separately and named as test-selected
  (`utils/checkpointing.py:67`, `utils/reconstruction_baselines.py:119`).
- Target-only forecast scores are placed on their raw target intervals and
  overlap aggregation uses cover counts (`utils/scoring.py:117`).

## 4. Proposed implementation

### P0. Make model semantics trustworthy

1. Repair real/imaginary channel packing and its exact inverse. Exercise multiple
   batch sizes, channels, odd/even frame counts, and STFT/iSTFT identity paths.
2. Introduce an unrestricted convolutional latent output for SIGReg-trained
   encoders. Preserve the old architecture as an explicit legacy definition.
3. Retain the agreed one-STFT feature pyramid, with explicit FP32 transforms
   under AMP. Do not introduce multiple FFT/window resolutions.
4. Implement the agreed attention Q/K/V roles and latent transpose output;
   document positional coordinates and future-guide capacity. A raw head must invert the
   effective backbone×encoder stride and output exactly the target raw span.
   A latent refinement head must preserve target token count. The current
   mirrored head can already serve a raw decoder when configured, but its
   automatically mirrored depth is not necessarily the requested small head.
5. Persist the input-access contract: reconstruction guide reads current
   observations; future guide reads observed future patterns. No target labels
   enter either action path. Do not introduce future values as a direct copy
   route without a deliberate experiment definition.
6. Declare mask placement for training and validation. Report fixed-corruption
   validation loss separately from uncorrupted deployment-style score behavior.
   Ensure preflight checks do not alter subsequent RNG-dependent experiments.

The agreed model contracts are:

- Guided attention uses future-derived Q and paired historical K/V. Q has one
  position per future latent token; historical K/V keep their own matching
  positions and may have a different length. No history-to-future-length
  resampling is needed to make K/V legal attention inputs.
- The predictor's small transpose module maps attention output to exactly
  `[B,D_target,T_target]`. Proposed minimal implementation: a stride-1,
  kernel-3, padding-1 ConvTranspose1d from attention value width to target latent
  width, with unrestricted output. A separate raw decoder is optional.
- Default both LeWM presets to latent-only prediction: `w_recon: 0`, no raw
  head, KL latent loss. Keep latent width independent of raw sensor count.
- The auxiliary-predicted soft mask drives the first model in training,
  validation and scoring. Known injected masks supervise its BCE only; they
  must not teacher-force predictor token placement in the default experiment.
- Derive nonempty history/future spans from the complete input at approximately
  **2:1**, with the split rounded to the effective feature stride. Default to
  stride 1 for the main experiment; retain valid strided configurations.
- Keep training prediction/target latents exactly aligned; interpolate scalar
  score maps during scoring only, if required by raw-time resolution. Compute
  L1 discrepancies on the latent grid first by default, reduce channels, then
  interpolate the resulting scalar map to the raw target span.

The first model's clean target is the uncorrupted view of the same window.
The engine retains that clean view before corruption; the auxiliary sees only
the corrupted current view. Known corruption locations supervise the auxiliary
but do not directly drive token replacement. During scoring, no fresh
corruption is injected: the observed input supplies both the current view and
the residual-reference latent, with the auxiliary proposing the soft mask.

Retain the existing KL loss definition, the symmetric divergence:
`p = softmax(prediction / tau, dim=channels)`,
`q = softmax(target / tau, dim=channels)`,
`d = 0.5 * sum_channels(p * log(p/q) + q * log(q/p))`, default `tau = 1`.
Compute with FP32 log-softmax for stability. Average `d` across batch and time
for prediction loss. Neither target
branch nor conditioning branch is detached. This objective ignores a shared
offset added to all channels at one token, while the default L1 score remains
sensitive to that offset. This is the user's deliberate loss/score combination.
Only use `d` as the score when the user explicitly selects KL scoring.

The confirmed default totals are `KL + BCE + 0.1*SIG_current +
0.1*SIG_target` for the auxiliary model and `KL + 0.1*SIG_current +
0.1*SIG_target` for the guided model. All weights are configurable. No default
raw term is included. Replacing the prediction loss must not change auxiliary
supervision, regularization weights or scoring mode implicitly.

The latent-prediction loss and latent score share exact-shape validation and
per-token discrepancy definitions but have independent selectors:

| Selector | Per-token discrepancy over latent channels D | Training reduction |
| --- | --- | --- |
| `kl` | Symmetric channel-softmax KL above, default temperature 1 | Mean over batch and tokens |
| `l1` | `mean_D(abs(prediction - target))` | Mean over batch and tokens |
| `mse` | `mean_D((prediction - target)^2)` | Mean over batch and tokens |
| `l2` | Proposed: `norm_2(prediction - target) / sqrt(D)` (RMS residual, matching existing L2 scoring) | Mean over batch and tokens |

The L2 convention is the sole new semantic detail awaiting final review. For
training, implement its norm so a perfect match has finite, zero gradient;
naively differentiating `sqrt(mean(error^2))` at zero can produce NaN. Test
all four loss choices independently of score choices, including default KL/L1,
finite gradients through both operands, zero residuals and FP32 scoring.

### P1. Reuse an encoder without changing its behavior

Extend the existing transfer helper at its current seam rather than adding a
second model-construction route. Read a versioned encoder description from the
source checkpoint, reconstruct it or reject conflicting destination options,
then load its state strictly. The description includes constructor type/options,
activations, kernels/dilations/strides, normalization, dropout, latent width,
feature ordering and normalization provenance.

Support source roles explicitly: a standalone source `encoder`, and (only if
requested) an effective source feature path containing `backbone + encoder`.
Copying the latter's downstream `encoder` alone is not raw-input-compatible.
Freeze transferred backbones by default; retain a clearly configured fine-tuning
option. Confirm feature equality in eval mode, not just weight equality.

Extend the AE assembly and forward path to support the two agreed LeWM-to-AE
routes: pretrained backbone plus identity encoder, or pretrained encoder with
`freeze_encoder` configurable and false by default. In the backbone route the
backbone remains frozen by default, consistent with the forecast reuse route;
fine-tuning is explicit. Frozen modules stay eval across `train()` calls, and
the decoder continues to receive gradients. Derive decoder widths/strides from
the complete effective feature path, including when its encoder is identity.
Both configurations remain ordinary reconstruction AEs without a latent
prediction or action loss.

Resolve sources per job using collection, series identity, experiment and
version. An explicit source checkpoint remains supported. Missing or mismatched
sources fail before a destination run starts. Do not silently fall back to
random backbone weights. Persist source identity and checkpoint fingerprint.
Use best-validation source weights, or best-training-loss source weights when
validation is unavailable, consistent with the agreed checkpoint policy.

Full resume restores its own backbone/encoder/optimizer/scheduler and does not
require the source file. Compare task metadata before resume/score; reject
semantic changes to crops, roles, strides, feature ordering or output domains.
Allow only documented nonsemantic overrides, such as extending epoch count.

### P2. One dataset selector planner and one trainer

Introduce a small planning interface that returns concrete training/evaluation
jobs. Jobs call the existing `train.run`; dataset selection must not create a
parallel training implementation.

Proposed CLI for the final review:

```bash
./venv/bin/python train.py --config_exp <experiment.yml> --dataset psm
./venv/bin/python train.py --config_exp <experiment.yml> --dataset smd --fname machine-1-1.txt
./venv/bin/python train.py --config_exp <experiment.yml> --dataset smd --fname all
./venv/bin/python train.py --config_exp <experiment.yml> --dataset smd --fname all-one
./venv/bin/python train.py --config_exp <experiment.yml> --dataset tsb-ad-u --fname <series.csv>
./venv/bin/python train.py --config_exp <experiment.yml> --dataset tsb-ad-m --file-list <manifest.csv>
```

`--fname` and `--file-list` are mutually exclusive. A manifest is selected
explicitly so it cannot be confused with a TSB series CSV. Validate collection,
root, files, duplicates, training prefixes, feature schemas, and source
checkpoints before training. Preserve the existing documented TSB manifest
format: required `file_name`, containing filenames relative to the selected
collection root; optional `train_end` when it cannot be inferred. Preserve list
order, accept unrelated metadata columns, and reject blank or missing entries,
nonexistent series and duplicate resolved paths. This matches the official-list
column documented in `configs/tsb_ad/README.md:62` and the surviving benchmark
CLI rather than inventing a new incompatible filename column.

PSM selection must clear an inherited SMD `data.file`, not merely omit a CLI
argument. TSB-U/M get distinct output namespaces and collection metadata while
sharing the existing reader. Allocate a fresh model/optimizer/calibrator for
each independent job; model widths may differ between TSB jobs.

Retain boundary-safe pooled SMD windowing. Add evaluation jobs for every member
using the shared checkpoint; calibrate and evaluate without concatenating
machines into one event timeline. Confirmed protocol: per-machine normalization
and thresholds, individual metrics plus macro summaries, with one shared model.

Write a run manifest with resolved config, seeds, dataset/source identities,
per-job status, checkpoint paths and metric paths. Preflight every selection
before training. After preflight, record individual runtime failures and
continue other jobs. Resume unfinished jobs without retraining completed jobs.
Update or remove references to obsolete runners; do not restore deleted legacy
code as an incidental dependency.

Expose bounded concurrent job execution, proposed as `--jobs N` (or the
equivalent config setting), default 1. Values greater than 1 enable parallel
jobs. Use independent worker processes, per-job model/optimizer/RNG state and
distinct output directories; the parent process owns the suite manifest.
Allow the explicit concurrency cap on a shared configured device, including
CUDA, rather than forcing one job per GPU. Record out-of-memory/runtime failures
per job without corrupting successful jobs. Do not add automatic retries that
silently alter batch size or experiment settings.

### P3. Trustworthy train/validation/test reporting

1. Fit normalization on optimization-training rows and reuse it unchanged for
   validation, clean calibration and test. Fix synthetic test normalization.
2. Use disjoint temporal validation when the source can hold both splits.
   Otherwise use the agreed validation-unavailable/best-training fallback
   below; reject inputs too short to produce even one legal training window.
   Preserve machine boundaries in joint splits. Invalidate caches when sources
   or the preprocessing contract change.
3. Log every unweighted loss term, configured weights and weighted total once.
   Verify total numerically and gradient paths independently. Sample weighting
   already exists and should be retained. SIGReg is batch-dependent, so its
   epoch summary is an average of batch statistics, not one full-dataset test.
4. Make validation corruption and SIGReg projections reproducible for checkpoint
   comparison. Keep validation RNG separate from training; preserve RNG states
   for deterministic resume when that guarantee is selected.
5. Use the
   full available clean training series for short-series calibration, as the
   user requires. On test data always report both clean-train-calibrated and
   test-oracle-threshold results, labeled separately. Oracle test labels may
   choose a diagnostic test threshold but never tune the model, preprocessing,
   calibrated threshold or validation-selected checkpoint. Keep the requested
   checkpoint set: rolling last checkpoint; best validation-loss checkpoint,
   or best training-loss checkpoint when validation is unavailable; and best
   test-F1 checkpoint separately for each evaluated aggregation strategy.
   Additional checkpoint-selection metrics are outside this specification.
   Select each strategy's best-test checkpoint using its ordinary F1 under
   the oracle threshold; attach both calibrated and oracle results. These are
   explicitly test-selected diagnostic checkpoints and do not replace the
   primary best-validation or best-training checkpoint.
   If no disjoint validation can fit, train and calibrate on all training rows,
   explicitly mark validation unavailable, and use best training loss for the
   primary best checkpoint. Never relabel training loss as validation loss.
6. Preserve window, non-overlap-window and timestep evaluation with correct
   target-span coverage, unscored-region exclusion and boundary-safe summaries.
   TSB evaluates the test suffix after the declared training prefix. Calibrated
   and oracle results use the same covered positions.
7. Record architecture/task/preprocessing metadata sufficient to reproduce
   score-only evaluation and transfer; update stale model/entry documentation.

## 5. Acceptance experiment after design agreement and implementation

The final smoke run is **not yet passed**. Initial probes establish execution
and specific defects, not anomaly-detection quality. Do not treat low synthetic
loss or a favorable tiny-test F1 as evidence of benchmark performance.

Use deterministic multichannel periodic/AR signals for clean train and held-out
validation, and a separate test with known spike, shift and frequency anomalies.
Retain one-channel and several-channel cases, multiple temporal lengths and
unequal forecast history/target lengths. Use local temporary output directories;
never replace `datasets/` or `results/` symlinks.

Required small experiment matrix:

| Run | Checks |
| --- | --- |
| AE | Forward, short training, finite component losses, held-out validation, score reload |
| VAE | Posterior shapes, stochastic training, deterministic mean-based scoring, MSE+beta*KL |
| Masked frequency LeWM | Correct spectral roundtrip; latent-only default with KL; all enabled loss terms/gradients; auxiliary-predicted masks in training and scoring |
| Guided LeWM without pretrained backbone | Future Q/history K/V; transpose output exactly matches target latent shape; target-only scores and raw-time interpolation |
| Guided LeWM from trained masked encoder | Exact source features; frozen weights/buffers; trainable destination gradients |
| Guided LeWM from trained AE encoder | Same checks, including correct source activation behavior |
| AE from trained LeWM backbone plus identity encoder | Exact effective source features and raw decoder geometry; frozen backbone stays eval and unchanged while decoder trains |
| AE from trained LeWM encoder | Trainable default and explicitly frozen modes; verify parameter and buffer behavior plus decoder gradients |

Use approximately 2–3 epochs and a few batches per epoch for plumbing checks;
use a fixed-batch overfit check separately when demonstrating optimization.
Assert exact objective summation, sample-weighted summaries including a partial
last batch, finite gradients and parameter updates, and no validation updates
to parameters or BatchNorm buffers. If gradient accumulation is requested,
add an independent effective-batch equivalence check.

Conditioning checks must verify both code paths and learned use: perturb/shuffle
the guide with history fixed, shuffle history with guide fixed, and compare
ablation results. At zero-initialized FiLM, an initially zero guide gradient is
expected; check its effect after the modulation projections have updated.
No guide ablation should be interpreted as a causal-physics demonstration.

Save/reload validation-selected weights and reproduce predictions, thresholds
and scores. Resume a short interrupted run without the original source file.
Reject incompatible source architecture, feature schema and task metadata.
Exercise validation-unavailable fixtures and confirm separate last,
best-training-loss, and per-aggregation best-test-F1 checkpoints. Verify bounded
parallel jobs have isolated outputs and RNG state and that a failed job does
not invalidate successful jobs or their resumability.

Generate PSM/SMD/TSB-U/TSB-M format fixtures and verify the exact selector/job
matrix, including SMD `all` versus `all-one` and TSB CSV lists. These fixtures
test format handling; follow with a small real-data sample per collection to
check installed paths and actual files. No full benchmark sweep is required
for smoke acceptance.

Deliver architecture summaries, resolved configs, train/validation component
curves, calibrated/oracle-separated test metrics, score coverage, transfer
provenance, and a concise pass/fail report with limitations.

## 6. Interview design tree

### Confirmed round 1

| Question | User decision |
| --- | --- |
| Q1: Future access | Yes: observed future is available to the guided action during scoring; this is delayed conditional comparison. |
| Q2: Input agnosticism | Yes: variable temporal lengths, fixed sensor schema per checkpoint. |
| Q3: Frequency resolution | Yes: one STFT and a multiscale learned pyramid only; no additional spectral resolutions. |
| Q4: Transfer | Yes to reconstruction-LeWM→forecast and AE→forecast. Also require LeWM→AE, either backbone plus identity encoder or pretrained AE encoder with user-enabled freezing; AE encoders train by default. |
| Q5: Learning recipe | Yes: shared encoder, gradients through both views, dual SIGReg, unrestricted latent outputs. Initial L1 choice is superseded by Q13: default latent prediction is KL. |
| Q6: Validation/calibration | Yes to held-out validation, train-only normalization and fixed validation corruption. For short series use the whole available training series for calibration. Test evaluation must include both calibrated and oracle threshold results. Q12 settles validation-unavailable behavior. |
| Q7: Joint SMD | Yes: one shared model, per-machine normalization/thresholds/results, macro summary. |

### Confirmed round 2

| Question | User decision |
| --- | --- |
| Q8: Attention | Yes: future-pattern Q, paired historical K/V. |
| Q9: Forecast output | Latent prediction, not raw reconstruction. Transpose output matches future latent length and width. Raw sensor width can differ. Prefer stride 1; use scoring interpolation at coarser strides. Optional separate transpose head for raw reconstruction. |
| Q10: Mask action | Yes: predicted soft mask in training and scoring; synthetic mask is auxiliary supervision only. |
| Q11: Crop ratio | Shape-derived and stride-aligned, but use approximately 2:1 history/future rather than 5:3. |
| Q12: Short series/checkpoints | Yes to training/calibrating on all available training rows without fabricated held-out validation. Keep last, best-validation or best-training when validation unavailable, and best-test-F1 checkpoint per aggregation strategy. Other checkpoint metrics are outside current scope. |
| Q13: Objective | Both models are LeWM latent predictors. No default raw reconstruction. Default latent loss is KL. Q20 confirms weights and configurable prediction losses. |
| Q14: Transfer source | Yes to matched dataset/series/experiment/version, explicit override and destination-resume precedence. Resolve to validation-best, with best-training fallback consistent with Q12. |
| Q15: TSB scope | Yes: test suffix only; both threshold protocols on identical covered positions. |
| Q16: Suite execution | Yes to preflight, record-and-continue runtime failures and resume unfinished jobs. Also require a configurable parallel-job toggle/cap; sequential remains the default. |

### Confirmed round 3

| Question | User decision |
| --- | --- |
| Q17: First model target | Same window: corrupted view is current state, clean view is future state in the chosen LeWM terminology. |
| Q18: Loss versus score | KL is the default loss; L1 is the default score. Configure them independently. Do not force matching geometries. |
| Q19: Test checkpoint selection | Yes: ordinary oracle F1 per aggregation strategy, both threshold protocols recorded, primary validation/train selection separate. |
| Q20: Weights and losses | Yes to prediction weight 1, auxiliary BCE weight 1 where applicable, and both SIGReg weights 0.1. Make latent reconstruction/prediction loss configurable among KL, L1, MSE and L2. |

### Final shared-understanding review

All experiment choices above are recorded. Proposed Q21: preserve the existing
L2 score convention for the new L2 loss, namely RMS over channels per token,
then mean over batch/time for training. This distinguishes L2 from MSE and
keeps its value normalized across latent widths. Final user confirmation of
this convention and the consolidated specification completes the interview.

The proposed minimal latent refinement is one length-preserving stride-1,
kernel-3 ConvTranspose1d with unrestricted output. Compute scalar errors on the
latent grid first, then linearly interpolate the score map to the raw target
span when necessary. Resolve source checkpoints to validation-best, falling
back to best-training when validation is unavailable, consistent with Q12.
Keep raw reconstruction opt-in: training a new raw decoder requires an
explicitly enabled raw loss; both default LeWM presets remain latent-only.
These proposed defaults are included in the final shared-understanding review.
Crop rounding, feature-grid position encoding, concurrency configuration and
checkpoint naming can use documented engineering defaults in that review.
Optimizer gradient accumulation is not currently requested; objective summation
and epoch loss aggregation are required and will be checked in the smoke experiment.

The user requested `/grill-me` after the initial check. The installed
[grilling skill](/home/manosbampis/.codex/skills/grilling/SKILL.md) says:
“Do not act on it until the user confirms you have reached a shared understanding.”
Accordingly this is a reviewable draft; implementation and final smoke acceptance
follow the interview, as the user requested.
