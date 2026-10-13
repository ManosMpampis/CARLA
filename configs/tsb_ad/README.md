# TSB-AD with native experiments and logging

`benchmark_tsb_ad.py` runs the existing models on TSB-AD data through their
native trainers. `experiments.yml` inherits `../psm/experiments.yml`, including
AE, VAE, time/frequency LeWM encoders, auxiliary variants, reconstruction, and
cross-attention heads. Selecting a head automatically includes its encoder
dependency. Input channels are inferred separately for each series. Existing
SMD/PSM training and evaluation commands remain available.

The complete downloaded benchmark has **not** been run as part of this
integration. Start with a dry run and one series before launching the full suite.

## Data and selection

Download and extract TSB-AD-U or TSB-AD-M using the
[official dataset instructions](https://github.com/TheDatumOrg/TSB-AD#-get-started).
The source checkout does not contain the full datasets. Point `--dataset-root`
at the extracted CSV directory; preserve this repository's `datasets/` and
`results/` symlinks. These commands run from the CARLA repository root:

```bash
TSB_DATA=/absolute/path/to/TSB-AD-M
TSB_SOURCE=/absolute/path/to/TSB-AD
TSB_FILE=001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv
TSB_OUTPUT="$PWD/results/benchmarks/tsb_ad/ae-smoke"
```

`TSB_SOURCE` is an optional checkout of the official GitHub repository. Its
`Datasets/File_List/` directory contains `TSB-AD-M-Tuning.csv`,
`TSB-AD-M-Eva.csv`, `TSB-AD-U-Tuning.csv`, and `TSB-AD-U-Eva.csv`, alongside
full lists. The U evaluation and U evaluation-full lists select different
sets. Use the tuning lists for configuration changes, then freeze the
configuration for evaluation. The manifest column must be `file_name`.
See the [official lists](https://github.com/TheDatumOrg/TSB-AD/tree/main/Datasets/File_List).

Choose `--files <relative filenames...>` or `--file-list <manifest.csv>`.
Omitting both discovers all CSVs recursively under the dataset root, so use
an extracted dataset directory rather than the source checkout as that root.

```bash
./venv/bin/python benchmark_tsb_ad.py \
  --dataset-root "$TSB_DATA" --files "$TSB_FILE" \
  --experiments ae/default --output-root "$TSB_OUTPUT" \
  --version benchmark-v1 --epochs 1 --dry-run

./venv/bin/python benchmark_tsb_ad.py \
  --dataset-root "$TSB_DATA" --files "$TSB_FILE" \
  --experiments ae/default --output-root "$TSB_OUTPUT" \
  --version benchmark-v1 --epochs 1 --device cuda
```

`--frameworks ae vae` selects whole framework families. `--experiments` selects
exact inherited keys, for example `lewm_encoder/time` or
`lewm_encoder/time/cross_attention/observed_input`. Omitting both selects the
full inherited suite. To run every inherited experiment on the official M
evaluation list:

```bash
./venv/bin/python benchmark_tsb_ad.py \
  --dataset-root "$TSB_DATA" \
  --file-list "$TSB_SOURCE/Datasets/File_List/TSB-AD-M-Eva.csv" \
  --output-root "$PWD/results/benchmarks/tsb_ad/m-evaluation" \
  --version m-full --epochs 300 --device cuda --dry-run

./venv/bin/python benchmark_tsb_ad.py \
  --dataset-root "$TSB_DATA" \
  --file-list "$TSB_SOURCE/Datasets/File_List/TSB-AD-M-Eva.csv" \
  --output-root "$PWD/results/benchmarks/tsb_ad/m-evaluation" \
  --version m-full --epochs 300 --device cuda
```

## Split, geometry, and alignment

Each CSV needs numeric feature columns followed by one binary `Label` column.
The `_tr_N_1st_M.csv` filename declares the first `N` rows as training data.
Normalization fits that entire prefix. Its final 20% is held out for validation
and threshold calibration, and the remaining prefix trains the model. Labels
never change the split, normalization, losses, or checkpoint selection. Some
official files contain anomalies inside their declared prefix; the final
report records `prefix_anomalies` as a diagnostic. Scoring and evaluation use
the **entire** CSV, including the prefix, following the official semisupervised
benchmark protocol. Invalid labels, missing values, and infinite features are
rejected instead of dropping rows and shifting the timeline.

The default `--window-policy shrink` shortens a requested window to fit both
training and validation, rounding down to a multiple of the encoder's total
stride. Default cross-attention half splits round to twice that stride to keep
the token count even. It caps the training stride at the resolved window and
reduces the batch size when fewer training windows are available. Model minimum
lengths still apply, with a larger minimum for frequency predictors. `--window`,
`--stride`, `--val-fraction`, and `--batch-size` set the requested values.
`--window-policy strict` rejects an incompatible window rather than shrinking
it; stride and batch caps still apply. The dry run prints each resolved window,
batch size, output directory, and selected checkpoint path.

Overlapping score maps are averaged by coverage. LeWM token scores are mapped
back to input timesteps. Cross-attention can leave an initial context prefix
without predicted targets. To export one finite score per timestamp, internal
gaps are linearly interpolated and leading/trailing uncovered positions extend
the nearest available score. This applies to all uncovered positions; it does
not create additional model predictions. `scores.npz` retains the original
`cover_counts`, starts, and ends, and `metrics.json` records the fill policy and
coverage. Inspect these artifacts when comparing a predictive head to a model
that directly scores every position. Decision traces show covered model outputs
before the benchmark's interpolation of uncovered positions.

## Logs, checkpoints, and results

Outputs default to `<configs/env.yml root_dir>/benchmarks/tsb_ad`. An explicit
`--output-root` keeps a benchmark or tuning variant in its own directory. For
the AE smoke command above, the native run directory is:

```text
<TSB_OUTPUT>/tsb_ad/ae/default/benchmark-v1/<TSB_FILE>/jepa_ae/
```

Each run retains native training logs and TensorBoard events, the resumable
checkpoint, validation-selected weights, and resolved configuration. AE/VAE
write `epochs.jsonl`, `last.pth.tar`, `best_validation_loss.pth.tar`, and
`tensorboard/`; LeWM and its heads retain `checkpoint.pth.tar`,
`model.pth.tar`, and their native logger events. Training-time test monitoring
is disabled for this protocol. Benchmark scoring adds:

- `benchmark_scores.npy`: the full-length, aligned score vector.
- `evaluation_inputs.npz`: labels, first-channel values, and calibrated predictions
  for later evaluation in another Python environment.
- `scores.npz`, `calibration.json`, and `metrics.json`: detailed scoring and
  prefix-validation threshold diagnostics.
- `benchmark_run.json`, `resolved_config.yml`, and `benchmark_result.json`:
  resolved configuration, input hash, run identity, and outcome.
- `benchmark.txt` and benchmark TensorBoard events; `decision_trace.html` and
  `decision_trace.json` replay up to 256 windows unless `--no-trace` is supplied.

The suite writes `<output-root>/benchmark_runs/<version>/summary.json` after
each experiment, preserving other experiments when a subset is resumed. It also
exports upstream-layout scores to `benchmark_scores/<algorithm>/<series>.npy`
within that version directory. Failed runs retain `benchmark_error.txt`; other independent
experiments continue unless `--fail-fast` is set. A failed encoder blocks its
dependent heads. The command exits nonzero if any run fails.

## Official metrics in a separate environment

Training defaults to `--metrics deferred`, so TSB-AD is optional in CARLA's
Python 3.14 environment. The training output already includes calibrated F1,
precision, and recall using a threshold from the prefix's validation tail.
Official benchmark metrics use upstream's oracle threshold searches and are
reported separately. They do not choose training checkpoints or calibration
thresholds.

Upstream documents Python 3.11/3.12 support and pins dependencies that differ
from this training environment. Follow its installation instructions in a
separate environment; avoid installing its full requirements into `venv/`.
For example, where Python 3.11 is available:

```bash
python3.11 -m venv "$HOME/.venvs/tsb-ad-metrics"
TSB_EVAL_PY="$HOME/.venvs/tsb-ad-metrics/bin/python"
"$TSB_EVAL_PY" -m pip install TSB-AD

"$TSB_EVAL_PY" evaluate_tsb_ad.py \
  --run-dir "$TSB_OUTPUT/benchmark_runs/benchmark-v1"
```

Alternatively pass `--tsb-ad-root "$TSB_SOURCE"` to load upstream metric code
from that checkout; its dependencies must still be present in the evaluator
environment. The evaluator imports no CARLA models or Torch training stack.
It writes `tsb_metrics.json` beside each successful score export and
`benchmark_metrics/` CSVs under the selected run directory with the nine official
metric columns. The default
metric buffer uses upstream's period estimate from the full first channel;
`--sliding-window N` explicitly overrides it and is independent of the model
window.

Prefer `--run-dir <output-root>/benchmark_runs/<version>`: the evaluator follows
that version's summary to the corresponding experiment artifacts, even though
their directories live elsewhere under the output root. When training omitted
`--version` and `--fresh`, use `benchmark_runs/default`. Passing an entire output
root or an experiment directory instead recursively discovers exports; repeated
experiment/file pairs across versions are rejected rather than combined.
For the complete M run above:

```bash
"$TSB_EVAL_PY" evaluate_tsb_ad.py \
  --run-dir "$PWD/results/benchmarks/tsb_ad/m-evaluation/benchmark_runs/m-full" \
  --tsb-ad-root "$TSB_SOURCE"
```

`benchmark_tsb_ad.py --metrics tsb-ad` also evaluates immediately when the
training environment can already import official TSB-AD metrics. Add
`--score-only` to repeat score export from validation-selected weights without
training, using the same output root, version, files, and configuration.

## Resume and fine-tune

Reusing an output root and version resumes that experiment's complete saved
state. `--epochs` is a **total target**, so extending a one-epoch smoke run to
ten epochs trains the remaining nine. `--learning-rate` updates the learning
rate after restoring optimizer and scheduler state:

```bash
./venv/bin/python benchmark_tsb_ad.py \
  --dataset-root "$TSB_DATA" --files "$TSB_FILE" \
  --experiments ae/default --output-root "$TSB_OUTPUT" \
  --version benchmark-v1 --epochs 10 --learning-rate 0.0001 --device cuda
```

Input hashes and resolved configuration protect against accidentally resuming
a different experiment. Use a new `--version` or `--fresh` when changing losses,
architecture, training geometry, or other fixed settings. `--fresh` generates
a new version and prints its paths. Omitting both uses the stable `default`
version. Separate output roots keep variant metric summaries independent.

For a new optimizer and schedule initialized from existing model weights,
select one explicit experiment and one file, then supply `--initialize-from`.
The architecture and channel count must match the checkpoint. For a head,
supply its complete checkpoint, including the encoder; phase-one dependency
training and separate encoder initialization are skipped. Existing state in
the destination run takes precedence, so choose a new version/output root:

```bash
TSB_WEIGHTS="$TSB_OUTPUT/tsb_ad/ae/default/benchmark-v1/$TSB_FILE/jepa_ae/best_validation_loss.pth.tar"
./venv/bin/python benchmark_tsb_ad.py \
  --dataset-root "$TSB_DATA" --files "$TSB_FILE" \
  --experiments ae/default \
  --output-root "$PWD/results/benchmarks/tsb_ad/ae-finetune" \
  --version lr-1e4 --epochs 10 --learning-rate 0.0001 \
  --initialize-from "$TSB_WEIGHTS" --device cuda
```

Cross-attention heads normally keep their encoder frozen. To fine-tune the
encoder as well, use a new run with a complete matching head checkpoint and
`cross_attention_kwargs.freeze_encoder: false`:

```bash
cat > /tmp/tsb-cross-finetune.yml <<'YAML'
cross_attention_kwargs:
  freeze_encoder: false
YAML
TSB_CROSS_WEIGHTS=/absolute/path/to/matching/cross_attention/model.pth.tar
./venv/bin/python benchmark_tsb_ad.py \
  --dataset-root "$TSB_DATA" --files "$TSB_FILE" \
  --experiments lewm_encoder/time/cross_attention/observed_input \
  --output-root "$PWD/results/benchmarks/tsb_ad/cross-finetune" \
  --version full-finetune --epochs 10 --learning-rate 0.0001 \
  --initialize-from "$TSB_CROSS_WEIGHTS" \
  --overrides /tmp/tsb-cross-finetune.yml --device cuda
```

Changing the encoder-freezing flag is allowed for this model-only
initialization; other checkpoint task settings, including target, query source,
and crops, still must match. Reconstruction heads use the following override
to update all parameters instead of only the head:

```yaml
stage_c:
  mode: finetune
```

LeWM phase-one models already update their full model during training. These
options change which parameters update, while `--initialize-from` starts a new
optimizer and scheduler in every case.

`--overrides path.yml` merges a YAML mapping into every selected experiment.
Use it for native loss, model, optimizer, or evaluation settings; for example:

```yaml
optimizer_kwargs:
  weight_decay: 0.001
score_batch_size: 128
```

CLI options such as `--learning-rate` and `--window` override corresponding
fields in that mapping. Review the dry run and saved `resolved_config.yml`.
Use training diagnostics and the official tuning lists for these changes;
keep evaluation-list results for the final frozen comparison. The
[integration research note](../../specs/tsb-ad-integration-research.md) links
the upstream split, scoring, and metric contracts.
