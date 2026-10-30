# TSB-AD integration research

Reviewed the official TSB-AD `main` sources on 2026-10-12 and validated the metric
adapter against checkout `ed74348e7876c9268b7a737c77f2f5f56f7fa458`. The links
below are moving upstream references; record the installed TSB-AD version and
checkout commit in each benchmark run for reproducibility. This note
distinguishes upstream behavior from recommendations for this repository.

## Dataset contract

The benchmark reads CSVs with a header, removes rows containing missing values,
takes every column except the last as floating-point features, and reads the
named `Label` column as integer ground truth. A multivariate input therefore
has shape `(timestamps, channels)`. The training prefix length comes from the
third token from the end of the filename stem, corresponding to
`..._tr_<length>_1st_<position>.csv`.
[Source: multivariate benchmark runner](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/Run_Detector_M.py#L49-L58).

For example, the first multivariate evaluation entry is
`001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv`. The manifest also contains
`078_SMD_id_22_Facility_tr_500_1st_326.csv`; some filenames consequently place
the first anomaly before the declared prefix ends. This is evidence against
assuming prefix cleanliness solely from the filename; checking the actual
labels is a diagnostic, not permission to silently redefine the split.
[Source: official multivariate evaluation manifest](https://github.com/TheDatumOrg/TSB-AD/blob/main/Datasets/File_List/TSB-AD-M-Eva.csv).

The upstream directory provides separate U/M tuning and evaluation manifests,
full U/M manifests, and an additional U evaluation-full manifest. Their column
name is `file_name`. The M tuning list contains 20 entries; the M evaluation
list contains 180. The ordinary U evaluation list is a selected set and must
not be conflated with its evaluation-full sibling.
[Source: manifest directory](https://github.com/TheDatumOrg/TSB-AD/tree/main/Datasets/File_List),
[M tuning list](https://github.com/TheDatumOrg/TSB-AD/blob/main/Datasets/File_List/TSB-AD-M-Tuning.csv),
[U evaluation list](https://github.com/TheDatumOrg/TSB-AD/blob/main/Datasets/File_List/TSB-AD-U-Eva.csv).

## Evaluation protocol and alignment

Both official benchmark runners train semisupervised methods on the declared
prefix, then pass the **entire series** as the scoring input and evaluate
against the **entire label vector**. They do not slice evaluation to the suffix.
Unsupervised methods instead receive the full series for both fitting and
scoring. Metrics use an automatically estimated period from the first channel.
Scores are saved as a NumPy array per series. To reproduce the benchmark,
preserve one finite anomaly score per timestamp, with larger scores indicating
greater abnormality, and do not omit a forecast warm-up region without making
the alignment policy explicit.
[Source: univariate benchmark runner](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/Run_Detector_U.py#L45-L91),
[custom detector contract](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/README.md#develop-your-own-algorithm).

The period estimator squeezes its input, uses at most 20,000 samples, obtains
an autocorrelation function through lag 400, and chooses a local maximum.
Its ordinary fallback is 125; a chosen period above 300 also returns 125.
The runners invoke rank 1 on the full first channel. This metric buffer is
separate from a model's context/window length.
[Source: period estimator](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/utils/slidingWindows.py#L5-L49).

`get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250)`
returns nine named values: `AUC-PR`, `AUC-ROC`, `VUS-PR`, `VUS-ROC`,
`Standard-F1`, `PA-F1`, `Event-based-F1`, `R-based-F1`, and `Affiliation-F`.
The four curve metrics are threshold-independent; the other five accept an
explicit prediction vector. Omitting `pred` selects oracle thresholds.
[Source: metric entry point](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/evaluation/metrics.py#L2-L35).

The oracle behavior actually optimizes individual F scores using labels:
standard F1 searches precision-recall thresholds; range, affiliation, point
adjusted, and event metrics sweep candidate thresholds. These figures are
valid benchmark comparisons but cannot represent deployable threshold
selection. Supplied predictions bypass those threshold searches; point
adjustment itself still uses labels as part of the metric definition. Keep
oracle results separate from predictions produced by training-only calibration.
[Source: metric implementations](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/evaluation/basic_metrics.py#L192-L328).

## Extension and CLI behavior

The wrappers dispatch by looking up a function named `run_<model_name>` in
`TSB_AD.model_wrapper`. The semisupervised dispatcher passes the training
array, scoring array, then keyword hyperparameters; the unsupervised dispatcher
passes only the scoring array and keywords. Both wrappers catch exceptions
and return an error string, so callers must validate type, shape, and finiteness
before treating output as a successful run. A custom model should return a
one-dimensional array aligned to the scoring input.
[Source: actual dispatch implementation](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/model_wrapper.py#L6-L38).

Upstream recommends an estimator subclassing `BaseDetector` and exposing
fitting, scoring, and stored decision scores. For integration into
the official runner, add the model name to the correct supervision pool and
provide best-configuration entries in both the univariate and multivariate
hyperparameter dictionaries. Tuning-grid dictionary entries are optional.
The documentation illustrates separate supervision-suffixed runner names;
the dispatch implementation above determines the actual registered callable
name. Models are imported lazily.
[Source: extension guide](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/README.md#develop-your-own-algorithm).

The official `Run_Detector_U.py` and `Run_Detector_M.py` accept
`--dataset_dir`, `--file_lsit` (upstream typo), `--score_dir`, `--save_dir`,
`--save`, and `--AD_Name`. They read an evaluation manifest by default, use
seed 2024, write scores under `<score_dir>/<AD_Name>/<series-stem>.npy`, and
skip an existing score file. When metrics saving is enabled, the CSV starts
with filename and runtime, followed by the nine metric columns. These details
come from the runners rather than the tutorial module.
[Source: multivariate runner CLI and output](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/Run_Detector_M.py#L12-L91).

The tutorial entry point `python -m TSB_AD.main` instead accepts
`--filename`, `--data_direc`, `--save`, and `--AD_Name`. It rescales scores and
passes predictions computed from their mean plus three standard deviations.
Use the benchmark runners' protocol for comparable results, not this tutorial
threshold convention.
[Source: tutorial entry point](https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/main.py#L27-L62).

Official tuning constructs the Cartesian product of the chosen model's
hyperparameter grid and evaluates each configuration on the tuning manifest.
It uses the same training-prefix/full-series convention as evaluation.
[Source: multivariate tuning script](https://github.com/TheDatumOrg/TSB-AD/blob/main/benchmark_exp/HP_Tuning_M.py#L29-L79).

## Dependency boundary

The README recommends a Python 3.11 environment and states support through
Python 3.12. Datasets are distributed separately from GitHub as U and M ZIP
archives; a source checkout alone does not supply the complete benchmark.
[Source: installation and dataset instructions](https://github.com/TheDatumOrg/TSB-AD#-get-started).

Although package metadata specifies only a Python lower bound, upstream pins
NumPy below 2 and depends on many detector libraries, including Torch,
Transformers, tslearn, stumpy, and arch. Installing this complete requirements
set into CARLA's Python 3.14 environment is not a supported upstream setup.
Keep upstream dependencies optional; avoid changing CARLA's working environment
merely to obtain benchmark metrics.
[Source: requirements](https://github.com/TheDatumOrg/TSB-AD/blob/main/requirements.txt),
[package metadata](https://github.com/TheDatumOrg/TSB-AD/blob/main/setup.py#L22-L40).

Local verification: importing just the official metrics and period estimator
from that source checkout worked with this repository's existing Python 3.14
dependencies, without installing or downgrading anything. A synthetic example
produced all nine metrics, and the adapter matched a direct upstream metric
call exactly. This verifies that narrow source-import path, not installation
or compatibility of upstream's entire detector dependency set. The standalone
evaluator remains usable in a separate supported Python environment.

## Recommendations for this repository

These are implementation recommendations inferred from the contracts above:

- Load native TSB-AD CSVs directly, fit normalization and calibration on the
  declared prefix, and never pass labels into model training or adaptation.
  Record missing-row handling because removal changes the index timeline.
- Reuse the existing model registries, trainers, checkpoints, and logger for
  each experiment arm. Make benchmark orchestration an adapter around those
  components, so loss curves and debugging artifacts remain available.
- Export full-length `.npy` scores for upstream compatibility, plus a richer
  archive containing coverage, alignment policy, calibration, runtime, and the
  exact resolved experiment configuration.
- Invoke official metrics lazily from an installed package or explicit source
  checkout. Allow score export without TSB-AD, and permit later evaluation of
  those scores in a supported Python environment.
- Store oracle benchmark metrics separately from fixed train-threshold
  predictions and repository metrics. Declare any held-out-only report as an
  additional report, because it changes upstream evaluation scope.
- Keep tune and evaluation manifests distinct. Fine-tune failing configurations
  using tuning-series results or training diagnostics, then freeze the chosen
  configuration before reporting evaluation-series comparisons.
