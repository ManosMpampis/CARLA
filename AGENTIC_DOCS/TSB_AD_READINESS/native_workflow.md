# TSB-AD native dataset workflow

Updated 2026-10-30 at the user's request. This supersedes the original integration's
default full-series evaluation and disabled training-time test monitoring.

## Behavior

Each selected CSV is treated as an independent series, like each SMD machine.
The filename's `_tr_N_` boundary separates the training prefix from the test
suffix. The prefix is further divided using `val_fraction` into optimization
and held-out validation/calibration windows. Normalization fits only the
declared prefix; test labels and values never enter optimization or calibration.

`tsb_evaluation_scope: test` is explicit in the generic, U, and M manifests and
is the loader default. Both scheduled and final evaluation use only the suffix.
`tsb_evaluation_scope: full` remains an explicit option for the original official
full-series export extent; it does not disable monitoring.

`eval_every` is no longer overwritten by the benchmark planner. Default configs
use 10; 1 evaluates every epoch, and 0 explicitly disables test monitoring.
Enabled monitoring also evaluates the final epoch. The same native evaluation
functions used by PSM/SMD write console metrics, TensorBoard `Test_*` sections,
`test_evaluations.jsonl`, and window/timeseries/non-overlap CSV reports. Optional
external official metrics can remain deferred without affecting this monitoring.

Final `metrics.json` includes native evaluations for all three procedures.
Exported scores, labels, first-channel data, and traces share the selected
timeline. `evaluation_start` records its offset into the original CSV.

Default handoff remains selected by validation loss. Separate diagnostic
test-oracle snapshots behave as in SMD/PSM. No checkpoint or result files from
existing runs were changed. Start a new version after adopting the changed
split; run-identity validation prevents mixing old and new evaluation extents.

## Diagnosis and verification

A new regression test first failed because the test dataset included the entire
CSV. Inspection confirmed the benchmark planner forcibly set `eval_every=0`.
An existing partial edit had also left an undefined `benchmark` variable in
the AE/VAE logging path; that path now checks the config directly.

Passed checks:

- `tests/verify_tsb_ad_native_workflow.py`: real CLI training for all five model
  families on synthetic CSV data, prefix/suffix separation, scheduled test reports
  at epochs 2 and 3, TensorBoard/CSV output, validation-selected handoff, aligned
  suffix exports, and resume to epoch 4 without duplicated evaluation history.
- `tests/verify_tsb_ad_data.py`: split, normalization, label isolation, explicit
  full-series compatibility, and malformed inputs.
- `tests/verify_tsb_ad_training.py`: monitoring disabled explicitly, weights-only
  initialization, and optimizer/LR resume.
- `tests/verify_tsb_ad_benchmark.py`: all 36 variants with explicitly selected
  full-series export and disabled monitoring, including resume and warm start.
- `tests/verify_tsb_ad_metrics.py`, `tests/verify_evaluation_cadence.py`, and
  `tests/verify_test_epoch_reporting.py`.
- Real Genesis multivariate file: two CPU epochs of the frequency encoder with
  `eval_every=1` produced reports at epochs 1 and 2 and 12,165 aligned test-suffix
  scores starting at original CSV row 4,055. Temporary smoke overrides used a
  128-row window and reduced metric buffer; artifacts are under
  `/tmp/carla-tsb-native-real-m`. No full dataset run was started.

The user-modified model geometry, selected experiment entries, and unrelated
working-tree edits were preserved.
