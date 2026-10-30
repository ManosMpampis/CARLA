# TSB-AD branch integration and validation

Date: 2026-10-30. Source: `codex/tsb-ad-benchmark` at `621630c`.
Destination: `jepa`, merge parent `8200d57` (same implementation as `9ef876c`,
plus the original readiness notes). This supersedes the pre-integration verdict
in [readiness.md](readiness.md).

## Merged behavior

`benchmark_tsb_ad.py` runs the native trainers on TSB CSVs with inferred channel
counts and declared-prefix splits. It adapts windows for short prefixes,
disables training-time test monitoring, selects weights by validation loss,
and exports full-series continuous scores. Calibrated decisions and optional
official oracle metrics remain distinct. `evaluate_tsb_ad.py` can evaluate
exported scores in an upstream-compatible Python environment.

The TSB suite explicitly lists the source branch's 36 variants. It no longer
inherits the destination's currently selected PSM experiments, where most
encoders and baselines are disabled. Existing SMD/PSM selections remain as they
were before this merge.

## Conflict resolution

- Kept the destination's shared `Checkpoints`, current loss logging, TensorBoard,
  periodic/final snapshots, and legacy checkpoint loading.
- Added weights-only initialization and resume learning-rate overrides from the
  benchmark branch. Existing resume state takes priority over initialization.
- Benchmark resume checkpoints are saved every epoch, retain initialization and
  protocol metadata, and use `checkpoint.pth.tar` for every model family.
- Moved the added integration research document into `AGENTIC_DOCS/specs/`,
  following the destination's directory rename; fixed its documentation link.

## Standards review

No hard violations found in registry usage, frozen metrics, train-prefix
calibration, symlink handling, or plain-Python verification. One correctness
finding overlapped with the spec review: a file-level planning failure could
stop deferred evaluation of successful files. Fixed as described below.

## Spec review

Two actionable findings were corrected: deferred evaluation now skips expected
directory-less failed planning entries while still rejecting malformed completed
entries; the training verification now references the renamed baseline configs.
A regression assertion covers mixed successful and failed planning entries.
Post-merge verification also exposed the suite-inheritance incompatibility and
old checkpoint-name assertions, both reconciled with the destination behavior.

## Passed verification

- `tests/verify_tsb_ad_data.py`: CSV contract, alignment, splits, normalization,
  and invalid input handling.
- `tests/verify_tsb_ad_training.py`: AE/VAE training without constructing test
  datasets, validation-only checkpoint selection, initialization, and LR resume.
- `tests/verify_tsb_ad_metrics.py`: official API handoff, deferred evaluation,
  malformed inputs, and mixed planning-failure summaries, using controlled
  upstream fixtures. This is not a real upstream numerical-parity run.
- `tests/verify_tsb_ad_benchmark.py`: all 36 variants through train and score,
  full-length score alignment, traces, resume, warm start, and provenance.
- Existing `tests/verify_evaluation_cadence.py`,
  `tests/verify_evaluation_reporting.py`, `tests/verify_test_epoch_reporting.py`,
  and `tests/verify_score_cli.py` (all five score CLIs).
- Real-data AE training and scoring, one epoch each on CPU, window 128, stride
  32, batch size 8, trace disabled: U file
  `001_NAB_id_1_Facility_tr_1007_1st_2014.csv` and M file
  `001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv`. Scores were finite and matched
  label timeline lengths. Artifacts: `/tmp/carla-tsb-real-u` and
  `/tmp/carla-tsb-real-m`. These smoke runs do not establish model quality.

## Existing failures and limits

`tests/verify_dataset_suites.py` fails because the selected native manifest
omits `lewm_encoder/frequency_aux` and the temporary output root has no saved
source checkpoint. `tests/verify_reconstruction_baselines.py` expects the old
`best_test_window_oracle_f1_epoch_0001.pth.tar` filename. Both failures were
reproduced on a clean archive of pre-merge HEAD under
`/tmp/carla-tsb-premerge-baseline`; neither was introduced by this merge.

The full 1,070-file benchmark and real upstream metric numerical parity were
not run. Official metrics remain deferred in real-data smoke outputs. Follow
[the run guide](../../configs/tsb_ad/README.md) for training, file-list selection,
and optional evaluation with upstream dependencies.
