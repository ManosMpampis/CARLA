# TSB-AD readiness audit

Audited 2026-10-30 at commit `9ef876c`. This is an assessment, not an implemented adapter.

## Verdict

The data are installed and the current model families can accept one-channel inputs,
but the repository cannot run native TSB-AD CSVs yet. Integration belongs primarily
in the shared dataset, experiment configuration, suite selection, and evaluation
protocol. No model rewrite is indicated by the checks below.

## Local evidence

- `datasets/TSB-AD-U` contains 870 CSV files; `datasets/TSB-AD-M` contains 200.
  These are accessed through the existing datasets symlink; it was not changed.
- `data/jepa_dataset.py:JEPADataset` implements only `synthetic`, `smd`, and `psm`.
  Direct construction with `train_db_name=tsb-ad-u` and `tsb-ad-m` raises
  `ValueError: Invalid train dataset ...` before attempting file loading.
- The SMD branch expects separate `train/`, `test/`, and `test_label/` paths and
  headerless numeric files. The sampled TSB files have a header, feature columns,
  and a final `Label` column in a single CSV. Renaming dataset settings does not
  reconcile these formats.
- Reading all installed CSV headers shows U always has one feature. M has feature
  counts 2, 3, 7, 8, 9, 12, 16, 17, 18, 19, 25, 29, 31, 38, 51, 55, 66, 248.
  Existing manifests set 25 channels for PSM or 38 for SMD. The model configuration
  must match each TSB file; existing weights with a different channel count are
  not directly interchangeable.
- CPU inference with random one-channel inputs succeeded for all five builders:
  AE, VAE, LeWM time predictor, reconstruction, and cross-attention. Existing
  YAML architectures were used with only `model_kwargs.in_channels` changed to 1.
  This checks architecture construction/scoring, not training or TSB performance.
- Filename training prefixes range from 500 to 50,000 rows in each collection.
  With `wsz=1024` and `val_fraction=0.1`, 720/870 U files and 116/200 M files
  have fewer than 1,024 validation rows. Some prefixes are shorter than even one
  training window. `JEPADataset.__len__` assumes full windows and has no short
  series fallback. Window and split policy must be designed before bulk runs.
- `utils/suite_cli.py` supports per-file enumeration specifically for SMD and
  discovers `train/machine-*.txt`. There is no TSB suite/config or tuning/evaluation
  list handling.
- `utils/evaluation_metrics.py:score_metrics` already computes VUS-PR, VUS-ROC,
  PR AUC, and ROC AUC. Existing `eval_window_size` defaults to 100. Matching an
  official benchmark requires checking its metric implementation, buffer/window
  selection, evaluated timeline, and tuning split rather than assuming that the
  same metric names ensure identical results.
- The current checkout has `lewm.py`, `lewm_reconstruction.py`,
  `lewm_cross_attention.py`, `carla_ae.py`, and `carla_vae.py`. The
  `carla_jepa.py` command in AGENTS.md is absent in this checkout.

## Required integration

1. Add a native TSB loader to the shared dataset path: parse the filename training
   boundary, exclude labels from model inputs, fit normalization on the training
   prefix, preserve the full evaluation timeline, and define missing-row handling.
2. Resolve input channels per file before constructing models and isolate each
   file's checkpoints and outputs. Continue using existing registries/builders.
3. Define explicit window/validation behavior for short prefixes and short series;
   verify predictor/crop/FFT requirements when reducing windows.
4. Add U/M manifests and a file/list runner, preserving official tuning/evaluation
   membership when reporting benchmark comparisons.
5. Preserve the repository's calibrated decisions and reports while adding an
   explicitly matched benchmark evaluation path over continuous timestep scores.
   Verify full timeline coverage, particularly cropped predictor outputs.
6. Verify one real U and one real M file through train, resume, and score before
   attempting collection-wide execution.

See [official_protocol.md](official_protocol.md) for upstream protocol evidence.
No application code, configs, datasets, or result artifacts were changed by this audit.
