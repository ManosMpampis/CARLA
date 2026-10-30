# Model and training design audit for SMD window detection

Written 2026-10-29. This appendix audits the current working-tree source. It explains mechanisms and limitations; the main report's saved-run measurements establish which mechanisms are plausible in particular experiments. A mechanism is not proof of a cause. No implementation, checkpoint, or result was changed.

## What the model is actually learning

The scientific objective is separation of anomalous from normal **windows**, with a usable normal-only threshold. Most training objectives optimize something else: restoration/prediction of normal signals or representations, plus a representation regularizer. Good normal reconstruction is useful only when anomalies are reconstructed sufficiently worse. Low training loss alone does not establish that contrast.

| Family | Actual learned task | Potential strength | Specific limitation to investigate |
|---|---|---|---|
| AE | Reconstruct normal input, minimize MSE | Direct telemetry error and simple control | A flexible model can reconstruct anomalous input too; validation MSE does not directly optimize window F1 or the upper normal-score tail |
| VAE | Same reconstruction plus beta-weighted posterior KL | Regularized information bottleneck | KL and MSE may trade off without improving anomaly separation; sampled training vs posterior-mean inference are different regimes |
| LEWM phase one | Restore clean shared-encoder latents from injected input; optional injected-mask prediction and FiLM | Learns invariances to controlled corruption and temporal/spectral structure | Synthetic corruption can differ from real faults; teacher-forced training actions differ from self-proposed scoring actions; default KL geometry differs from L1 scoring |
| Frozen-encoder reconstruction | Train raw-input reconstruction head on normal windows | Tests whether phase-one features preserve useful telemetry | Frozen encoder cannot recover fault distinctions it already discards; normal reconstruction remains a proxy for separation |
| Observed cross attention | Use context keys/values and **observed target-crop queries** to reconstruct input/features/query | Can detect incompatibility between adjacent windows without predicting an entirely unseen future | Conditional reconstruction can preserve or explain away a fault already visible in its query; it is not forecasting |
| Context cross attention | Build query from context only and predict target crop | Actual past-only comparison | Benign regime changes can be surprising; faults already present in context may become predictable; difficult forecasts need not be anomalies |

AE/VAE facts: [models/ae_baseline.py:37](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/ae_baseline.py:37), [objective:59](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/ae_baseline.py:59). VAE training samples only in training, and validation/scoring use the posterior mean: [utils/reconstruction_baselines.py:217](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/reconstruction_baselines.py:217), [models/ae_baseline.py:49](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/ae_baseline.py:49). Reconstruction uses clean normal windows and freezes encoder/aux by default: [lewm_reconstruction.py:107](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/lewm_reconstruction.py:107), [dense training:146](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/lewm_reconstruction.py:146).

### A concrete loss/scoring mismatch in phase one

The phase-one presets use symmetric **channel-wise softmax KL**: [frequency_aux config:34](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/configs/lewm_encoder/frequency_aux/phase1.yml:34), [time config:29](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/configs/lewm_encoder/time/phase1.yml:29). The actual loss applies softmax over latent channels, while scoring takes raw latent absolute differences: [losses/lewm.py:37](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/losses/lewm.py:37), [models/lewm.py:501](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm.py:501).

Mathematically, adding the same scalar to every predicted channel at a token leaves its softmax unchanged, so it leaves this KL unchanged; the raw L1 residual changes. Therefore, better validation KL need not imply a stable L1 score scale or better calibrated window F1. This is an exact property of the implemented objectives, not an empirical attribution of any particular bad run. SIGReg constrains the encoder distributions, but it does not make KL and raw L1 identical objectives.

The AE/VAE presets similarly train with MSE (plus VAE KL) while defaulting to L1 scoring: [AE config:18](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/configs/baselines/ae.yml:18), [VAE config:18](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/configs/baselines/vae.yml:18). This mismatch is weaker than the softmax invariance: MSE and L1 both penalize amplitude errors, but weight large errors differently.

### Train actions, inference actions, and injected anomalies

LEWM phase one uses the known injected mask to replace corrupted latent positions with a learned token **in training**. In evaluation, the auxiliary proposes a soft mask; without an auxiliary, the action is all zero. The predictor must therefore perform under a different action regime when scored: [models/lewm.py:410](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm.py:410), [eval branch:454](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm.py:454), [scoring:489](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm.py:489). Validation follows the inference action path, which helps expose the discrepancy, but it still uses synthetic injections.

Injected examples span 10–90% of an input window in the presets. The implementation chooses temporal spans randomly, modifies a random subset of channels, and includes compression, trend, scaling and shapelet operations: [utils/masking_steered.py:30](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/masking_steered.py:30), [operators:38](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/masking_steered.py:38), [channel selection:95](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/masking_steered.py:95). These are controlled training corruptions, not measured coverage of real SMD fault types. Inference may fail on subtle correlation changes or realistic regime changes even after excellent injected-mask performance. That remains a hypothesis until fault segments and per-channel evidence are inspected.

### What frequency auxiliary and FiLM add

The frequency predictor applies an STFT to learned latent features, processes a time-frequency grid, then inverts the STFT: [models/lewm.py:304](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm.py:304). Its auxiliary is trained to identify injected spans and also supplies features for additive/multiplicative FiLM steering: [models/lewm.py:166](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm.py:166), [losses/lewm.py:63](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/losses/lewm.py:63). Time predictors have an analogous FiLM path: [models/lewm.py:189](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm.py:189).

This can improve restoration where the auxiliary recognizes a corruption. Conversely, auxiliary mistakes affect both what gets masked and how restoration is conditioned. Compare `aux`, `aux_no_film`, and no-aux arms on the same machines to separate those contributions. A stronger downstream head after `frequency_aux` establishes that its transferred representation is useful for that head; it does not establish that the frequency predictor itself is the best deployment detector.

### Cross-attention shortcuts and moving targets

The source explicitly distinguishes observed from context queries: [models/lewm_cross_attention.py:235](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm_cross_attention.py:235). There is no direct Query residual around attention: [attention computation:75](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm_cross_attention.py:75). Nevertheless, observed target information controls retrieval, so a narrow bottleneck alone cannot prove that anomalies will be rejected.

`target=features` uses the frozen encoder's second-crop features; `target=input` uses telemetry; `target=query` uses the clean **trainable Query convolution** output: [target construction:247](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm_cross_attention.py:247). Query-target loss has no detach: [losses/lewm_cross_attention.py:6](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/losses/lewm_cross_attention.py:6). Both the predictor and target representation can therefore adapt toward an easier shared task. SIGReg resists distribution collapse, but does not guarantee that information relevant to every real fault is retained. A nonzero aggregate latent variance does not rule out loss of one fault-sensitive direction.

Observed presets use full-window feature extraction; nearby context and target features can share noncausal convolutional receptive fields. Context presets use separate extraction and reject full extraction: [feature extraction:225](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm_cross_attention.py:225), [constructor guard:121](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm_cross_attention.py:121). The first is a legitimate observed reconstruction task, but its results must not be described as isolated future forecasting.

New positional presets add fixed sinusoids to Q/K only for attention addressing. They do not alter the content targets or SIGReg inputs: [models/lewm_cross_attention.py:75](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm_cross_attention.py:75). Improvements would support the hypothesis that temporal addressing was insufficient. They would not by themselves establish better anomaly specificity or generalization to other window lengths.

## Validation, checkpoint selection, and threshold stability

### Which checkpoint is deployable evidence?

The ordinary LEWM trainer validates prediction loss in eval mode and writes `model.pth.tar` when that loss improves: [utils/trainer.py:119](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/trainer.py:119), [selection:163](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/trainer.py:163). It selects only `pred_loss`, excluding auxiliary loss and SIGReg from the selection value. Their effects still influence training.

It also saves a separate `jepa_model_best` path based on the test hook's returned F1: [utils/trainer.py:170](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/trainer.py:170). Critically, this hook returns **timeseries calibrated F1**, not window F1: [utils/reconstruction_baselines.py:322](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/reconstruction_baselines.py:322). Therefore a filename containing `best` cannot establish window-best performance. It is test-selected and unsuitable as unbiased evidence of checkpoint selection without labels.

AE/VAE use the full validation objective MSE + beta KL, save `best_validation_loss.pth.tar`, and explicitly keep distinct test-best F1/VUS checkpoints: [utils/reconstruction_baselines.py:408](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/reconstruction_baselines.py:408). The suite's default comparison loads validation-selected weights; configured checkpoint sweeps can override that: [utils/experiment_suite.py:302](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/experiment_suite.py:302), [selection metadata:403](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/experiment_suite.py:403).

Finding a test-oracle epoch is an upper-bound diagnostic. A more useful question is how far the validation-selected epoch lies below that bound, and whether the plateau is broad enough that a label-free selection rule could find it. Repeatedly choosing architectures, epochs or quantiles using these test scores makes subsequent test claims exploratory even if the final inference threshold is train-only.

### Validation noise is not all optimizer instability

The LEWM and cross-attention runners create separate corruption collators for train and validation: [lewm.py:116](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/lewm.py:116), [lewm_cross_attention.py:92](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/lewm_cross_attention.py:92). Validation collators draw fresh spans and perturbations from global NumPy randomness on every call. Hence adjacent validation points include changing corruption difficulty as well as changing weights. SIGReg also samples fresh directions, but Trainer's validation selection excludes its value: [losses/sigreg.py:40](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/losses/sigreg.py:40), [utils/trainer.py:136](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/trainer.py:136).

A globally seeded run is reproducible under the same execution sequence; it does not imply that the same validation windows receive the same injected faults each epoch. A fixed held-out corruption bank would make checkpoint comparisons easier to interpret. Train/eval loss gaps are also affected by dropout, BatchNorm, and the action/sampling differences above, so they are not automatically overfitting evidence.

### Nominal holdout is useful but not fully independent

The SMD loader fits its standardizer on the **entire provided train file**, then cuts a nominal tail for validation: [data/jepa_dataset.py:120](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/data/jepa_dataset.py:120). There is no test-data normalization here, but the holdout contributes to preprocessing statistics. The same tail is used for checkpoint validation and clean-score quantile calibration: [lewm_cross_attention.py:126](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/lewm_cross_attention.py:126). This is practical normal-only calibration, not an independent estimate of false alarms under future nominal regimes. A deployment evaluation should separate optimization, checkpoint validation, calibration and final labeled audit intervals where data permits.

Current common suite defaults use a single seed, one 10% nominal tail, a high 0.995 quantile and overlapping windows: [configs/experiments.yml:7](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/configs/experiments.yml:7). Historical runs may use other settings; inspect saved metadata instead of assuming the latest manifest describes them. Windows overlap heavily and therefore provide less independent information about a high quantile than their raw count suggests. A threshold can look precise numerically while being sensitive to a few contiguous nominal episodes.

### Stability measurements should answer different questions

1. **Representation/optimization stability:** late prediction-loss variability, catastrophic spikes, collapse diagnostics. Useful, but only indirect deployment evidence.
2. **Decision stability:** late calibrated window F1 variability, recall and false-positive changes, and how many late evaluation epochs remain close to each run's test-oracle F1. Test labels make these retrospective diagnostics.
3. **Score-scale stability:** robust relative movement of calibrated and oracle thresholds. Do not compare raw thresholds across latent/input/Query targets with different units. A pure positive affine score change leaves ranking intact and can move raw thresholds dramatically.
4. **Threshold transferability:** calibrated-to-oracle F1 gap at the same checkpoint, oracle threshold's percentile within clean scores, and width of thresholds that retain near-best F1. A small raw threshold drift can still be damaging if that useful interval is narrow.
5. **Reproducibility:** repeat across seeds and nominal calibration intervals. Epoch stability within one run cannot establish this.

The saved evaluation cadence determines what can be known. If metrics appear every ten epochs, any reported first near-oracle epoch is the first **observed** one. Brief excursions between evaluations are unmeasured. A final score archive generally measures threshold sensitivity at its saved checkpoint, not at every historical checkpoint.

## Targeted next experiments and their falsifiable questions

These are proposed experiments, not conclusions established by the audit.

| Suspected limitation | Targeted comparison | Observation supporting the suspicion |
|---|---|---|
| Phase-one loss/score mismatch | Same initialization/data, L1 training+L1 score vs KL training+L1 score; or KL-consistent inference | Better validation-to-decision association and reduced calibrated/oracle gap without sacrificing window VUS |
| Unstable validation corruptions | Fixed validation corruption bank plus an independent clean-score tail | Much smoother selection values at identical weights; fewer checkpoint-selection reversals |
| Query target learns an easy representation | Match observed input/features/query arms and inspect fault-segment scores | Query loss improves while normal/anomaly score separation degrades; fixed-domain targets retain separation |
| Aux proposals miss real faults | Score full windows vs frozen aux-cropped inference and inspect proposal recall | Missed anomaly windows have low/empty proposals despite strong full reconstruction residuals |
| Frozen features discard fault information | Same phase-two head with frozen encoder vs controlled finetuning | Fault-sensitive residual separation improves specifically where frozen-feature distances were small |
| Mean score dilutes short/sparse faults | Keep model/checkpoint fixed, compare mean vs predeclared top-k/channel-aware reductions | Gains concentrate in low anomaly-occupancy windows without a corresponding explosion of nominal alerts |
| Threshold tail fails to transfer | Freeze model; use disjoint nominal calibration intervals and test fixed false-alarm targets | Large interval-to-interval threshold/F1 differences with unchanged ranking |

For aux-crop specifically, unproposed positions score exactly zero, and an empty proposal yields an all-zero map: [models/recon_model.py:95](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/recon_model.py:95). Current suite scoring deliberately disables that path, matching training-time full reconstruction monitoring: [utils/experiment_suite.py:295](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/experiment_suite.py:295). Thus an `aux_crop` configuration name does not prove its suite window results used cropped inference; inspect saved run/evaluation metadata.

## Empirical audit: short occupancy inside a positive window

This section uses existing score archives from four complete base families, **28 machines each**, and does no inference or training. The reproducible [audit_window_occupancy.py](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/audit_window_occupancy.py) writes [window_occupancy.json](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/window_occupancy.json) and [window_occupancy.csv](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/window_occupancy.csv). The JSON preserves all per-machine values and source archive paths.

For each saved scored window, anomaly occupancy is the fraction of timesteps annotated anomalous within its actual `start_idxs:end_idxs` span. A prefix sum of timestep labels computes this efficiently. The script asserts that `occupancy > 0` exactly matches saved `window_labels`, and that strict thresholding at the saved calibrated threshold exactly matches `window_predictions`. All **112 archives** pass. No timeseries or non-overlap scores enter the calculations; timestep labels supply only occupancy within the already defined window task.

All numbers below are equal-machine means. Positive-window Spearman correlation asks whether higher anomaly occupancy is associated with a higher score **among positive windows only**, avoiding the trivial distinction between zero-occupancy normal windows and positive ones.

| Base family | Positive-window score/occupancy Spearman | Calibrated recall for occupancy ≤10% | Eligible machines | Recall for occupancy >50% | Eligible machines |
|---|---:|---:|---:|---:|---:|
| Frequency phase one | 0.272 | 0.659 | 28 | 0.811 | 11 |
| Frequency auxiliary phase one | 0.308 | 0.699 | 28 | 0.818 | 11 |
| AE | 0.336 | 0.769 | 28 | 0.909 | 11 |
| Frequency auxiliary observed Query | 0.330 | 0.621 | 28 | 0.730 | 16 |

Because the high-occupancy subgroup has fewer eligible machines, a paired comparison uses **the same machines for both bins**. Low/high recall is 0.683/0.811 for frequency (11 machines), 0.663/0.818 for frequency auxiliary (11), 0.779/0.909 for AE (11), and 0.551/0.730 for observed Query (16). Within these paired groups, recall is lower for low-occupancy positive windows on average.

Missed positive windows also have lower average occupancy than correctly detected positive windows on **19/24** eligible frequency machines, **15/20** frequency-auxiliary machines, **16/18** AE machines, and **19/23** observed-Query machines. Only machines with at least one calibrated false negative and one true positive qualify. The paired mean false-negative minus true-positive occupancy difference is −0.050, −0.046, −0.061 and −0.094 respectively. These are absolute occupancy-fraction differences.

This supports the practical concern that a mean score over a long window may struggle with a short fault inside an otherwise normal window. It does **not** prove that mean pooling causes the misses: short-overlap windows may contain less severe fault portions, be near fault boundaries, or differ in other properties. The correlations are moderate, not close to 1. A fixed-checkpoint pooling ablation and inspection of fault boundaries would be the causal follow-up.

### Why a high window F1 can still describe a weak detector

The any-anomaly window labels make the positive-window prevalence much higher than the mean occupancy of those windows. Across machines, frequency, frequency auxiliary and AE have mean window prevalence **0.420**, while their positive windows contain on average only **0.106** anomalous timesteps as a fraction of the window. Observed Query has prevalence **0.263** and positive-window mean occupancy **0.168**. The latter evaluates the target half of the input; its task geometry differs. These are means of per-machine quantities, not a pooled dataset prevalence, and the occupancy mean must not be multiplied by macro prevalence to infer total anomaly fraction.

An always-positive classifier has F1 `2p/(1+p)` when positive-window prevalence is `p`. Its equal-machine F1 is **0.557** for the first three families and **0.395** for observed Query. The saved oracle F1 is exactly equal to that trivial baseline (within 1e-10) in these cases:

| Family | Machine | Oracle F1 = always-positive F1 |
|---|---|---:|
| Frequency phase one | machine-2-7.txt | 0.8624 |
| Frequency phase one | machine-3-3.txt | 0.9397 |
| AE | machine-1-8.txt | 0.7523 |
| AE | machine-2-4.txt | 0.8939 |
| AE | machine-2-7.txt | 0.8624 |

Equality is a metric comparison; it does not claim that the saved oracle makes precisely the same individual predictions as the always-positive classifier. It establishes that its best F1 provides **no F1 improvement** over raising an alert on every evaluated window for that machine. In particular, a 0.940 oracle F1 can coexist with poor usefulness when almost all evaluated windows are positive. The audit found no exact equality cases for frequency-auxiliary phase one or observed Query.

Cross-family differences in this table are descriptive, not an architecture-controlled benchmark: scored spans differ, checkpoint epochs differ, and the study has not resampled fault episodes or training seeds. These results should complement the main report's F1/VUS/threshold ranking, not replace it with an occupancy-only ranking.
