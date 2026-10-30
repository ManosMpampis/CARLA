# SMD experiment suite

`experiments_smd_all.py` trains and evaluates the same experiment variants as
the PSM suite, using SMD's 38 channels. `experiments.yml` inherits the experiment
list from the PSM manifest and replaces the dataset, filename, and channel count.
Shared Python runners take dataset settings from the config.

```bash
# Preview all experiments on the manifest's default machine (machine-1-1.txt)
./venv/bin/python experiments_smd_all.py --dry-run

# Train and evaluate all experiments on one machine
./venv/bin/python experiments_smd_all.py --fname machine-1-1.txt

# Select machines or discover every machine under datasets/SMD/train
./venv/bin/python experiments_smd_all.py --machines machine-1-1.txt machine-1-2.txt
./venv/bin/python experiments_smd_all.py --all-machines

# Evaluate saved selected checkpoints without training
./venv/bin/python experiments_smd_all.py --all-machines --score-only

# Short run of a selected framework
./venv/bin/python experiments_smd_all.py --fname machine-1-1.txt --frameworks ae --epochs 1
```

Other options match `experiments_psm_all.py`: `--experiments` (exact keys),
`--manifest`, `--config-env`, `--version`, `--fresh`, `--device`,
`--cpu-threads`, and `--fail-fast`. `--dataset-root` overrides `datasets/SMD`;
the directory must contain `train/`, `test/`, and `test_label/` with matching
headerless machine files. All-machine mode trains each machine separately;
it does not join machines into a shared training corpus.

Stable output examples:

```text
results/smd/ae/default/machine-1-1.txt/jepa_ae/
results/smd/lewm_encoder/time/phase1/machine-1-1.txt/jepa_time_predictor/
results/smd/lewm_encoder/time/cross_attention/observed_input/machine-1-1.txt/jepa_cross_attention_observed_input/
results/smd/summaries/machine-1-1.txt/Test_timeseries_summary.csv
```

An explicit `--version myrun` inserts `myrun/` immediately before the machine
filename (and before the machine filename in summary paths). Machine names
isolate checkpoints, calibration, scores, and summary rows.

For multiple checkpoints and input/crop cases, edit `evaluation.yml` and run:

```bash
./venv/bin/python evaluate_suite.py --manifest configs/smd/evaluation.yml --dry-run
./venv/bin/python evaluate_suite.py --manifest configs/smd/evaluation.yml
```

This uses the machine configured in `experiments.yml`; the generic evaluator
has no default dataset. See `README_EVALUATION.md` for crop units and reports.

## Group experiments below the dataset

Set `common.experiment_version` in a suite manifest to insert a directory
immediately after the dataset. For example, `configs/smd/logging_check_10e.yml`
sets `experiment_version: logging_check_10e` and runs five models for 10 epochs
on `machine-1-1.txt`:

```bash
./venv/bin/python experiments_smd_all.py \
  --manifest configs/smd/logging_check_10e.yml --version run1 --fail-fast
```

Example output paths:

```text
results/smd/logging_check_10e/ae/default/run1/machine-1-1.txt/jepa_ae/
results/smd/logging_check_10e/lewm_encoder/frequency/phase1/run1/machine-1-1.txt/jepa_frequency_predictor/
results/smd/logging_check_10e/summaries/run1/machine-1-1.txt/
```

`experiment_version` groups the architectures; `--version` selects a run inside
that group. Without `--version`, the extra run directory is omitted. Leaving
`experiment_version` unset preserves the original layout. Shared encoder
handoffs, resume lookup, suite summaries, and default evaluation-suite outputs
all follow the group. Explicit evaluation `output_dir` still takes precedence.
Existing result directories are preserved; this setting selects a new location.
