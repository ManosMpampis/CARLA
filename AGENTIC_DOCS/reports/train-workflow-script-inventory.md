# Scripts outside the `train.py` workflow

This inventory covers the current working tree. The workflow is `train.py` for AE, VAE, masked LeWM, and guided LeWM, including training, validation, checkpoint resume, encoder transfer, periodic test evaluation, and `--score`. The audit traced imports and conditional dispatch, then ran `tests/verify_training_entry.py` to confirm the modules loaded by the synthetic end-to-end run. Optional dataset and scoring paths were checked from source. The inventory includes repository Python and JavaScript scripts, including tests and report helpers. It excludes generated output, the environment, and inline JavaScript in HTML.

## Repository root

| Script | Where it is used |
| --- | --- |
| `carla_ae.py` | Earlier AE command, still exercised by the score CLI checks. |
| `carla_vae.py` | Earlier VAE command, still exercised by the score CLI checks. |
| `lewm.py` | Earlier LeWM command and phase runner, still called by the experiment suite and legacy wrappers. |
| `lewm_reconstruction.py` | Earlier reconstruction phase runner, still called by sweeps and the experiment suite. |
| `lewm_cross_attention.py` | Earlier cross-attention phase runner, still called by sweeps and the experiment suite. This is distinct from `models/lewm_cross_attention.py`. |
| `experiments_cross_attention.py` | Standalone legacy cross-attention sweep. |
| `experiments_psm.py` | Standalone PSM sweep. |
| `experiments_psm_all.py` | Standalone PSM experiment suite. |
| `experiments_rec.py` | Standalone legacy reconstruction sweep. |
| `experiments_smd_all.py` | Standalone SMD experiment suite. |
| `benchmark_tsb_ad.py` | Standalone TSB-AD benchmark orchestration. |
| `evaluate_tsb_ad.py` | Standalone evaluator for saved TSB-AD benchmark runs. |
| `evaluate_suite.py` | Standalone evaluation of checkpoints across configured cases. |
| `aggregate_lewm_metrics.py` | Aggregates saved LeWM metrics across machines. |
| `aggregate_smd_summaries.py` | Compares saved SMD machine summaries. |
| `aggregate_tsb_ad_summaries.py` | Aggregates saved TSB-AD metrics across series. |
| `oracle_threshold.py` | Standalone threshold diagnostic for completed runs. |
| `print.py` | Historical experiment plotting script. |

## `data/`

- `SMD.py` is the older SMD dataset class. The unified loader reads SMD in `jepa_dataset.py`.
- `custom_dataset.py` defines an older dataset class. The audit found no current caller.

## `losses/`

- `lewm_cross_attention.py` supplies the legacy Q/K/V loss. It is still registered and can be selected explicitly. Its default SIGReg terms require outputs from the old attention model.
- `lewm_rec_ca.py` supplies the legacy Rec+CA loss and is still registered. It requires the old model's additional reconstruction outputs.

Neither legacy loss is selected by the shipped unified configs. The unified guided forecast uses the ordinary LeWM loss.

## `metrics/`

- `metrics.py` provides the earlier combined metric interface. `utils/reporting.py` and assertion scripts still reference it.
- `Matthews_correlation_coefficient.py` supplies MCC to that combined metric interface.

The unified evaluator still uses the VUS, affiliation, and point-adjusted F1 modules.

## `models/`

- `builders.py` is the compatibility re-export. The audit found no current Python caller in the repo.
- `action_planning.py` implements optional discrete-action planning. The composable model test and model documentation reference it.

## `utils/`

| Script | Where it is used |
| --- | --- |
| `evaluation_suite.py` | Standalone evaluation suite. |
| `evaluation_summary.py` | Writes cross-run and suite summaries. Per-run scoring uses `evaluation_artifacts.py`. |
| `reporting.py` | Earlier score-reporting implementation. The audit found no current Python caller. |
| `suite_cli.py` | Shared CLI for dataset experiment suites. |
| `timeseries_aggregation.py` | Separately tested aggregation interface documented in `AGENTIC_DOCS/Design/TimeseriesAggregator.md`. The unified runner uses `utils/scoring.py`. |
| `tsb_ad_benchmark.py` | Standalone TSB-AD benchmark orchestration. |
| `tsb_ad_metrics.py` | Adapter for the separate TSB-AD evaluation workflow. |

## `tests/`

All of these are run separately from training. `verify_training_entry.py` is the direct end-to-end check for the unified workflow.

- `check_scorer_handoff.py`
- `verify_arms.py`
- `verify_calibration.py`
- `verify_cross_attention_diagonal_mask.py`
- `verify_dataset_suites.py`
- `verify_decision_trace_ui.cjs`
- `verify_evaluation_cadence.py`
- `verify_evaluation_mapping.py`
- `verify_evaluation_report_ui.cjs`
- `verify_evaluation_reporting.py`
- `verify_evaluation_suite.py`
- `verify_evaluation_summary.py`
- `verify_experiment_group_paths.py`
- `verify_experiment_layout.py`
- `verify_lewm_composable.py`
- `verify_lewm_cross_attention.py`
- `verify_lewm_cross_attention_tcn.py`
- `verify_lewm_frequency_steering.py`
- `verify_lewm_phase1.py`
- `verify_lewm_rec_ca.py`
- `verify_lewm_reconstruction.py`
- `verify_lewm_true.py`
- `verify_model_builders.py`
- `verify_non_overlap_window.py`
- `verify_oracle_calibration_quantile.py`
- `verify_pretrain_tracer.py`
- `verify_psm_auto_resume.py`
- `verify_psm_suite.py`
- `verify_reconstruction_baselines.py`
- `verify_score_cli.py`
- `verify_sigreg.py`
- `verify_stable_experiment_paths.py`
- `verify_stages.py`
- `verify_test_epoch_reporting.py`
- `verify_test_epoch_suite.py`
- `verify_timeseries_aggregation.py`
- `verify_training_entry.py`
- `verify_training_splits.py`
- `verify_tsb_ad_benchmark.py`
- `verify_tsb_ad_data.py`
- `verify_tsb_ad_metrics.py`
- `verify_tsb_ad_native_workflow.py`
- `verify_tsb_ad_planning.py`
- `verify_tsb_ad_summaries.py`
- `verify_tsb_ad_training.py`
- `verify_vus_window_config.py`

## `AGENTIC_DOCS/presentations/lewm_reconstructor/.build/`

- `build.mjs` builds the LeWM reconstruction presentation.

## `AGENTIC_DOCS/reports/smd_cross_attention_visual_audit/`

- `analyze_traces.py`
- `compare_matched_heads.py`
- `verify_trace_samples.py`
- `write_report.py`

## `AGENTIC_DOCS/reports/smd_window_analysis/`

- `analyze.py`
- `audit_thresholds.py`
- `audit_window_occupancy.py`
- `metric_semantics_probe.py`
- `summarize.py`
- `verify_vus_artifacts.py`
- `write_report.py`

## Files the unified workflow still needs

- `models/__init__.py` imports `models/lewm_cross_attention_tcn.py`, `models/lewm_rec_ca.py`, and `models/recon_model.py` when loading the registry. Their legacy classes are not constructed by the unified presets, but removing the files would break the import. `models/lewm_cross_attention.py` also defines the guided predictor that the unified forecast constructs.
- `data/PSM.py` and `data/original_dataset.py` are used when loading PSM. `data/tsb_ad.py` is used when loading TSB-AD.
- `utils/experiment_suite.py` supplies the shared `score_config` function used by `train.py --score`. The older suite runners in that file are separate call paths.
- `utils/decision_trace.py` and `utils/inference_benchmark.py` are used when the respective optional scoring settings are enabled.
