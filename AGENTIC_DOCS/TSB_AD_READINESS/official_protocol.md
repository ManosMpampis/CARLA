# Official TSB-AD protocol notes

Checked against the official repository's `main` branch on 2026-10-30.
These are upstream facts and integration implications, not evidence that the
local CARLA implementation already satisfies them.

## Data and split

- TSB-AD-U and TSB-AD-M are the univariate and multivariate collections.
  GitHub contains examples; full datasets are separate ZIP downloads. Names
  encode `[index]_[Dataset Name]_id_[id]_[Domain]_tr_[Train Index]_1st_[First Anomaly Index].csv`.
  [Dataset documentation](https://github.com/TheDatumOrg/TSB-AD/blob/main/Datasets/README.md).
- CSV files have headers. The official loaders call `pd.read_csv(...).dropna()`,
  use all columns except the last as float features, and read the `Label`
  column as integer ground truth. The training boundary is extracted as
  `filename.split('.')[0].split('_')[-3]`.
  [Multivariate runner](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/Run_Detector_M.py).
- The bundled NAB example has header `Data,Label` and filename
  `001_NAB_id_1_Facility_tr_1007_1st_2014.csv`.
  [Example CSV](https://github.com/TheDatumOrg/TSB-AD/blob/main/Datasets/TSB-AD-U/001_NAB_id_1_Facility_tr_1007_1st_2014.csv).
- Semisupervised dispatch receives `data_train = data[:train_index]` and
  `data_test = data` (the entire series). Unsupervised dispatch receives the
  entire series for fitting/scoring. Scores are evaluated against the entire
  label sequence and saved as `.npy`.
  [Univariate runner](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/Run_Detector_U.py).

Integration implication: training on the clean prefix and scoring only the
remaining suffix is a valid local experiment, but does not reproduce the
official full-series evaluation extent. Preserve both extents explicitly if
both are desired. Exclude `Label` from model inputs. Resolve missing-row
handling and the resulting prefix boundary consistently with upstream.

## Benchmark selection

- Official lists are CSVs with a `file_name` column. Separate U/M tuning and
  evaluation lists are provided; there is also `TSB-AD-U-Eva-Full.csv`.
  [File lists](https://github.com/TheDatumOrg/TSB-AD/tree/main/Datasets/File_List).
- Hyperparameter tuning enumerates a detector's grid on the tuning list;
  `HP_Tuning_M.py` defaults to `TSB-AD-M-Tuning.csv` and computes metrics using
  tuning labels. The evaluation runner defaults to `TSB-AD-M-Eva.csv` and
  uses the selected optimal detector settings.
  [Tuning script](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/HP_Tuning_M.py),
  [evaluation script](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/Run_Detector_M.py).

Integration implication: do not conflate tuning-series label use for
benchmark model selection with evaluation-series labels entering training or
calibration. Freeze settings before running the evaluation list.

## Scores and metrics

- The published custom-detector interface uses arrays shaped
  `(n_samples, n_features)` and returns one score per timestamp, with larger
  scores meaning more anomalous. Its documented `BaseDetector` subclass and
  registration steps apply to integration/submission inside the official
  TSB-AD repository; they are not necessary just to load the data in CARLA.
  [Custom detector instructions](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/README.md).
- Both official U/M experiment runners pass
  `find_length_rank(data[:, 0].reshape(-1, 1), rank=1)` as the metric's
  `slidingWindow`, and call `get_metrics` without a prediction vector.
  [Multivariate runner](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/Run_Detector_M.py),
  [univariate runner](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/Run_Detector_U.py).
- `find_length_rank` estimates a period from autocorrelation on at most the
  first 20,000 samples, searching lags through 400. Its rank-1 path selects
  the strongest local ACF maximum and falls back to 125 when selection fails
  or the period exceeds 300.
  [Period estimator](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/utils/slidingWindows.py).
- `get_metrics` returns AUC-PR, AUC-ROC, VUS-PR, VUS-ROC, Standard-F1, PA-F1,
  Event-based-F1, R-based-F1, and Affiliation-F. VUS uses the passed
  `slidingWindow`; defaults are `version='opt'` and `thre=250`. With
  `pred=None`, the source explicitly describes threshold-dependent metrics
  as using oracle thresholds.
  [Metric entry point](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/evaluation/metrics.py).

Integration implication: model context length and the official VUS tolerance
window are separate parameters. Keep existing clean-train calibrated
threshold metrics and label any separately reproduced oracle F1 metrics.
Matching metric names alone does not establish numerical comparability.
Check full-series score alignment, tolerance window, metric implementation,
and threshold convention before comparing to published TSB-AD values.

## Dependency scope

For dataset-only use, an adapter can consume CSVs and filename metadata with
CARLA's existing scientific Python stack; installing or porting all official
detectors is unnecessary. This is an integration recommendation derived from
the upstream loader and array contract, not a tested dependency guarantee.
