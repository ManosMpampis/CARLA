# TSB-AD-M experiment summaries

This folder holds derived reports from saved benchmark results. The reusable script is `aggregate_tsb_ad_summaries.py` at the repository root. Keep training code and model artifacts outside this folder.

Regenerate from the repository root:

```bash
./venv/bin/python aggregate_tsb_ad_summaries.py --version first_run
```

Each version directory contains three split directories (`all`, `tuning`, `eval`). Each split contains `non_overlap_summary.csv`, `window_summary.csv`, and `timeseries_summary.csv`, with one row per experiment and mean, sample standard deviation, and valid-value count for every requested metric. Matching `*_files.csv` exports retain individual series metrics and errors.

The default includes only filenames declaring a training prefix strictly greater than 500 steps. Membership comes from the dataset's all, Tuning, and Eva CSVs. Means weight each series equally. Failed/missing runs and missing/nonfinite values contribute no metric samples; blanks indicate no valid observations. Coverage counts remain visible. Singleton standard deviations are zero.

Metrics are read from the final saved `metrics.json`, with no model training, epoch selection, or threshold recomputation. Oracle metrics describe the stored oracle decision; they are not separately re-optimized for each F measure. Calibrated metrics retain the saved calibration decision. VUS and AUC values are independent of those decisions.

`file_inventory.csv` records split membership and exclusions; `failed_runs.csv` records eligible failed runs; `manifest.json` records source paths and aggregation semantics. CSV exports are ignored by the repository's existing `*.csv` rule, but remain available locally and can be regenerated.
