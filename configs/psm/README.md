# PSM experiment suite

Run every currently implemented framework and its configured arms:

```bash
./venv/bin/python experiments_psm_all.py --version psm_v1
```

Without `--version`, the launcher checks each experiment's run folders and
resumes the most recently modified training checkpoint (`last.pth.tar` for
AE/VAE, `checkpoint.pth.tar` for LEWM and phase two). Experiments without a
checkpoint start in a new timestamped run. Use `--fresh` to start new runs,
or `--version NAME` to select a specific run. `--dry-run` shows the selected
paths without training. Resuming restores model, optimizer, scheduler, and
epoch; `--epochs` is the total epoch target. Saved phase-one weights can also
supply dependencies whose entries are omitted from the manifest.

The default manifest, `configs/psm/experiments.yml`, includes 16 experiments:

| Framework directory | Experiments |
| --- | --- |
| `ae` | `default` |
| `vae` | `default` |
| `lewm_encoder` | time/frequency, each without auxiliary, with auxiliary steering, and with auxiliary without FiLM |
| `reconstruction` | `default`, `aux_crop` |
| `cross_attention` | observed/past-only Query × features/input/Query targets |

Deleted implementations and unimplemented design documents are excluded. All
entries use the existing training code. Source LEWM experiments run before
their dependent reconstruction/cross-attention arms and are trained once per
suite run. Their checkpoint paths are resolved automatically.

Standalone reconstruction uses `lewm_reconstruction.py` for `pretrain`,
`recon`, and `score`; cross attention uses `lewm_cross_attention.py` for both
training phases and scoring. Final model assembly lives in `models/builders.py`.

The suite uses PSM's 25 telemetry channels from `datasets/PSM/train.csv`,
`test.csv`, and `test_label.csv`. `common` sets shared data, window, seed,
calibration, and encoder settings. The supplied comparison uses a 128-step
input, encoder channels `[32,32]`, and strides `[1,1]` across frameworks. Its
epoch counts come from each existing base config. Change `common.epochs` for
one shared training duration, or put `epochs` in an entry's overrides.

The existing `experiments_psm.py` runs the reconstruction chain using the same
shared encoder hierarchy. See `configs/lewm_encoder/README.md` for inherited
configs, shared pretraining, and custom phase-two variants.

## Framework, experiment name, and run

These fields identify the framework, shared encoder, variant, and run:

| Field | Purpose | Examples |
| --- | --- | --- |
| `framework` | Select the framework branch | `vae`, `ae`, `lewm_encoder`, `reconstruction`, `cross_attention` |
| `phase1_experiment` | Name the shared LEWM encoder | `time`, `frequency_aux` |
| `experiment_name` | Name a hyperparameter/model variant within that framework | `default`, `lr_0.1`, `beta_0.5` |
| `--version` | Identify a particular run; reuse it to resume | `psm_v1`, `seed4_run2` |

To train two VAE experiments, keep the existing `vae/default` entry and add:

```yaml
- framework: vae
  experiment_name: lr_0.1
  runner: vae
  config: ../baselines/smd_vae.yml
  overrides:
    optimizer_kwargs:
      lr: 0.1
```

This example is also commented out in the manifest. The name is your label;
`lr_0.1` does not change learning rate by itself. `overrides` changes it.
Overrides merge recursively, so changing `lr` preserves `weight_decay`.
Config paths are relative to the manifest file. Names contain letters, digits,
underscores, dots, and hyphens. Phase-two names must be unique within each
phase-one encoder and framework branch. The same name under another encoder
is a separate experiment. Use its complete key for selection, for example
`lewm_encoder/time/cross_attention/context_input`.

The resulting layout groups both variants together:

```text
results/psm/
  vae/
    default/
      psm_v1/psm/jepa_vae/
    lr_0.1/
      psm_v1/psm/jepa_vae/
  ae/
    default/psm_v1/psm/jepa_ae/
  lewm_encoder/
    time/
      phase1/psm_v1/psm/jepa_time_predictor/
      reconstruction/default/psm_v1/psm/jepa/
      cross_attention/observed_input/psm_v1/psm/jepa_cross_attention_observed_input/
    frequency_aux/
      phase1/psm_v1/psm/jepa_frequency_predictor_time_annotation_steering/
      reconstruction/default/psm_v1/psm/jepa/
      reconstruction/aux_crop/psm_v1/psm/jepa/
  summaries/
    psm_v1/summary.csv
    psm_v1/summary.json
```

`tag_jepa` retains its existing meaning: it names the model directory at the
end of the path. `framework` and `experiment_name` control the grouping above
it. For LEWM descendants, `phase1_experiment` inserts the shared encoder
parent and the phase-two framework chooses its branch. The same fields work
in standalone model YAML files through
`utils/config.create_config`. Configs without `framework` retain their original
output paths, including existing configs that use `experiment_name` as a
descriptive display label.

Each experiment saves a `resolved_config.yml`, training checkpoints and
TensorBoard records, plus `metrics.json`, `calibration.json`, `scores.npz`, and
suite evaluation TensorBoard records. Reusing a run checks its resolved
configuration before resuming; extending epochs or changing runtime device,
AMP, workers, or score batch size is allowed. Use a new experiment name or
version for a changed model, loss, learning rate, or data/window configuration.

## Choosing what to run

```bash
# Show all experiments, output directories, and source checkpoints; write nothing
./venv/bin/python experiments_psm_all.py --version psm_v1 --dry-run

# Run only the VAE entries, including any variants you added
./venv/bin/python experiments_psm_all.py --version psm_v1 --frameworks vae

# Run one exact experiment; required source LEWM is added automatically
./venv/bin/python experiments_psm_all.py --version psm_v1 \
  --experiments lewm_encoder/time/cross_attention/context_input

# A short training run (scoring still evaluates the complete PSM test series)
./venv/bin/python experiments_psm_all.py --version psm_smoke --epochs 1

# Resume the same run
./venv/bin/python experiments_psm_all.py --version psm_v1

# Re-evaluate existing validation-selected checkpoints without training
./venv/bin/python experiments_psm_all.py --version psm_v1 --score
```

`--manifest` chooses another experiment list, `--config-env` chooses another
output-root config, and `--device` overrides the device. Optional
`--cpu-threads` controls Torch's CPU thread count. When `--version` is omitted,
a timestamp names the run. A custom PSM location can be set with
`common.dataset_root`; the CSV schema stays the standard PSM schema.

To change the source encoder, update the corresponding source LEWM entry and
its dependent entries consistently. `pretrained_experiment` identifies the
source as `lewm_encoder/<phase1 name>`; dependent `model_kwargs` must match
the source exactly. An existing external LEWM checkpoint can instead be set
through an entry's `overrides.pretrained_from`, omitting
`pretrained_experiment`.

## Results and selection

The runner prints a final table and saves CSV/JSON summaries. Each row includes
framework, phase-one name, phase-two experiment name, complete experiment key, run, status, calibrated time-series/window F1,
time-series VUS PR/ROC, scoring coverage, checkpoint and metric paths, elapsed
time, and any failure. Summaries update after every experiment. Running a
subset later updates its rows while preserving the other experiments' rows.

Every summary uses **validation-selected weights**: AE/VAE
`best_validation_loss.pth.tar`, or the Trainer's best-validation
`model.pth.tar` for LEWM/reconstruction/cross-attention. Thresholds are clean
training-tail quantiles. The final test labels do not select these checkpoints
or calibrated thresholds. The AE/VAE training code also saves its existing
diagnostic best-test checkpoints; those are excluded from the suite summary.
Reconstruction test-monitoring cadence is disabled in the supplied manifest.

All frameworks use the same window/time-series scoring and evaluation code.
Point-adjusted metrics have a separate `point_adjust_comparability` section;
saved predictions remain unadjusted. Test-oracle thresholds are explicitly
separate from calibrated thresholds in `evaluation` and are excluded from the
summary's F1 columns. Score reduction/model training still follows each arm's
own implementation.

The cross-attention arms score only the second half of each input. Their window
labels refer to that target half. Other frameworks score complete windows.
Coverage counts exclude unscored positions from calibration and time-series
evaluation; summaries expose output-window length and scoring coverage so the
different procedures remain visible. Observed Query is conditional
reconstruction; past-only Query is forecasting.

Failures are recorded with a traceback under the failing experiment, and
unrelated experiments continue. Dependants of failed sources are marked
`blocked`. The command exits nonzero if a requested experiment fails. Add
`--fail-fast` to stop at the first failure.

```bash
./venv/bin/python tests/verify_psm_suite.py
```

This assertion script exercises all 16 default experiments plus a named VAE
variant on temporary PSM-format CSV files, including checkpoint handoffs,
calibration/reporting, grouping, resume, score-only operation, configuration
guards, and failure handling. It writes no files to the real dataset/results
symlinks. It verifies execution and contracts, not full-dataset model quality.
