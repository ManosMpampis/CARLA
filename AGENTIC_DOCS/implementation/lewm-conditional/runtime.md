# Conditional experiment runtime

Use `train.py` for AE, VAE and the two conditional LeWM configurations. Dataset
selection expands into native training jobs; each job constructs its own model,
optimizer and RNG state. The default concurrency is one. Generated fixtures and
smoke outputs belong in temporary directories; installed dataset/results links
are never replaced.

```bash
./venv/bin/python train.py --config_exp configs/training/ae.yml --dataset psm
./venv/bin/python train.py --config_exp configs/training/lewm.yml --dataset smd --fname machine-1-1.txt
./venv/bin/python train.py --config_exp configs/training/lewm.yml --dataset smd --fname all --jobs 2
./venv/bin/python train.py --config_exp configs/training/lewm.yml --dataset smd --fname all-one
./venv/bin/python train.py --config_exp configs/training/ae.yml --dataset tsb-ad-u --dataset-root /path/TSB-AD-U --fname series_tr_1000_1st_1200.csv
./venv/bin/python train.py --config_exp configs/training/ae.yml --dataset tsb-ad-m --dataset-root /path/TSB-AD-M --file-list /path/list.csv --jobs 2
```

`--fname` and `--file-list` are mutually exclusive. PSM clears inherited SMD
filenames and takes no filename. SMD `all` trains one model per machine;
`all-one` trains one shared model with windows confined to each machine. TSB
manifests require `file_name`, optionally `train_end`; unrelated metadata columns
are accepted. List order is preserved. Paths stay relative to the collection
root; duplicates, blank rows, absent series and invalid prefixes fail preflight.
The U and M collections have separate output namespaces.

`--output-root /tmp/my-smoke` changes the result root without editing the env
file. `--dataset-root` changes the selected collection root. Add `--version`
for a new run. `--score` reloads the primary best snapshot, with optional
`--score_checkpoint`; it uses a separate resumable suite manifest from training.
`benchmark_tsb_ad.py` is an alias for this same CLI. Its former benchmark-only
flags and deleted orchestration modules are not restored.

Every selection is preflighted before any optimizer step. The parent owns the
JSON suite manifest under `<root>/<collection>/suites/<framework>/<experiment>/`.
It records resolved configs, seeds, artifact paths, worker IDs and statuses.
Runtime failures are recorded and other jobs continue. Repeat the same command
to resume unfinished jobs and skip completed jobs with surviving artifacts;
increasing epochs resumes those runs. `--jobs N` bounds independent spawn
workers, including on a shared CUDA device. A runtime failure never changes
batch size or experiment settings automatically.

## Training and evaluation

Normalization fits optimization-training rows only. Validation is a disjoint
temporal tail when both sides hold at least one window. Otherwise validation is
explicitly unavailable, optimization/calibration use all training rows and
primary checkpoint selection uses training loss. Pooled SMD applies this rule
independently to every machine; only members with a disjoint tail contribute
validation windows. Normalization caches include source/preprocessing identity.

Validation resets independent global and criterion-local RNG streams, including
seeded SIGReg projections, and restores training streams afterward. Loss terms
and the weighted total are logged with sample-count-weighted batch averages;
`resolved_config.yml` and checkpoint inference metadata record objective weights.
SIGReg summaries remain averages of batch statistics.

Every epoch writes rolling `checkpoint.pth.tar` and `last.pth.tar`. The primary
snapshot is `best_validation_loss.pth.tar`, or `best_train_loss.pth.tar` when
validation is unavailable; `model.pth.tar` aliases that primary selection.
Separate `best_test_<strategy>_oracle_f1.pth.tar` files are explicitly
test-selected diagnostic snapshots. Each contains both calibrated and oracle
results. `final.pth.tar` is the last epoch, independent of primary selection.

Evaluation calibrates thresholds on the full clean training series and reports
both calibrated and test-oracle thresholds on identical covered test positions.
TSB uses only the declared test suffix. The window, non-overlap-window and
timeseries strategies retain target-only forecast coverage. Final artifacts
include `metrics.json`, `calibration.json`, `scores.npz`, and score figures.
Pooled SMD evaluates/calibrates each machine independently and writes member
artifacts under `machines/`; the top-level report is a macro summary with
finite metric counts, never one concatenated event timeline.

## Encoder sources

Source roles accept an explicit checkpoint path, a path template containing
`{dataset}`, `{collection}`, `{fname}` and `{version}`, or a matched source:

```yaml
backbone:
  source:
    framework: lewm
    experiment: masked_frequency
    version: default
  freeze: true
```

For AE, the same backbone can precede `encoder: {name: identity}`. Alternatively
place `source` under `encoder` and choose its `freeze` flag. Source mappings
resolve the matching dataset/series checkpoint to validation-best, with
training-best fallback. Explicit `pretrained_from` remains available. Exact
feature architecture and provenance are embedded in destination checkpoints;
full destination resume does not require the source file. Semantic model,
feature-schema, runtime and preprocessing changes are rejected on resume.

Permanent acceptance checks are `tests/verify_conditional_models.py`,
`tests/verify_conditional_runtime.py`, `tests/verify_conditional_transfer.py`
and `tests/verify_conditional_suites.py`.
