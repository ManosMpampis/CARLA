# SMD window anomaly detection: experiment audit and practical shortlist

Follow-up, 30 October: the [observed-features visual audit](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/REPORT.md) examines all 28 full-series traces. It confirms useful fault peaks and competitive matched-head results, and separates calibration failures from duration and score-separation failures.

Analysis date: 29 October 2026. This is an exploratory audit of saved experiments, not a new training run or an independent final test. The extraction snapshot contains **1,019 selected-checkpoint reports**, **30,570 scheduled evaluation records**, **1,800,600 TensorBoard scalar records**, and **4,536 additional inference-geometry evaluations**. The main comparison contains **36 variants × 28 machines = 1,008 runs**, all seed 4. Eleven reports from two newer positional-encoding variants are incomplete and are discussed separately.

Only **`evaluation.window`** contributes to every performance ranking here. F1 is ordinary binary F1 with **no point adjustment**. The other evaluation protocols are excluded. Every machine has equal weight, irrespective of its number of overlapping windows. A family mean is the mean of per-machine metrics, not a pooled score across different score units.

## 1. What I would investigate first

**There is useful anomaly-ranking information, but the current clean-tail threshold is the largest obstacle to a reliably useful detector.** Stable optimization helps, yet it does not solve threshold transfer between nominal calibration and the test regimes.

- **For a stable engineering starting point:** keep **AE/default** and **frequency_aux / reconstruction / auxilary_added**. Their calibrated F1 is close to the leading mean, while late threshold movement and decision variation are much smaller than phase-one prediction. The reconstruction arm is the most stable full-span variant by the measured late F1 SD; AE is a simpler independent control.
- **For the highest current mean calibrated F1:** inspect **frequency phase one** (0.5201). It is a useful reference, but the approximately 0.0093 advantage over AE is too uncertain across these machines to establish a decisive winner.
- **For representation/ranking headroom:** inspect **frequency_aux_no_film phase one**: highest full-span mean oracle F1 (0.7622) and VUS-PR (0.7105). **frequency_aux** has the highest mean VUS-ROC (0.7083). Their calibration gaps and training behavior prevent treating these area-score wins as deployment wins.
- **For the target-half cross-attention task:** keep **frequency_aux / observed_query**, **frequency_aux_no_film / observed_query**, and **frequency_aux_no_film / observed_features**. They answer a different labeling task from full-span detectors. No completed context-query experiment was present in this snapshot, so this audit cannot judge past-only forecasting empirically.

None of these is established as deployment-ready. Their normal-window false-positive rates, failures on particular machines, and calibration gaps remain material. A low numerical loss or a high oracle F1 alone does not establish a useful alarm system.

## 2. What the scores actually show

### Full-span detectors: input 1,024, scored output 1,024, stride 10

| Experiment | Calibrated F1 | Oracle F1 | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- |
| frequency | 0.5201 | 0.7490 | 0.7039 | 0.7028 |
| frequency_aux | 0.5189 | 0.7562 | 0.7060 | 0.7083 |
| frequency_aux_no_film | 0.5082 | 0.7622 | 0.7105 | 0.7066 |
| ae/default | 0.5108 | 0.7308 | 0.6754 | 0.6759 |
| vae/default | 0.5002 | 0.7274 | 0.6708 | 0.6667 |
| frequency / recon / default | 0.5125 | 0.7264 | 0.6718 | 0.6755 |
| frequency_aux / recon / auxilary_added | 0.5053 | 0.7300 | 0.6759 | 0.6872 |

The three frequency phase-one variants are close. Frequency beats AE in calibrated F1 on 18/28 machines, but a paired machine bootstrap gives a mean difference of **+0.0093, 95% descriptive interval −0.0737 to +0.0887**. Frequency_aux minus frequency is **−0.0012**, interval **−0.0588 to +0.0479**. These intervals describe variability across these machines at this seed; they do not measure seed reproducibility, establish equivalence, or replace a held-out architecture-selection test.

### Target-half cross attention: input 1,024, scored output 512, stride 10

| Experiment | Calibrated F1 | Oracle F1 | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- |
| frequency_aux_no_film / CA / observed_query | 0.4207 | 0.6363 | 0.5642 | 0.7013 |
| frequency_aux / CA / observed_query | 0.4150 | 0.6404 | 0.5692 | 0.7059 |
| frequency_aux_no_film / CA / observed_features | 0.4060 | 0.6564 | 0.5647 | 0.6984 |

The target-half label is positive if **any anomaly appears in the scored half**. Full-span methods apply that rule to the entire 1,024 points. Across machines, average positive-window prevalence is approximately **0.420 for the full span versus 0.263 for the half span**. Their absolute PR and F1 values therefore measure different tasks. Rank heads within matched geometry before attributing the difference to architecture.

![Selected-checkpoint window F1 and VUS across candidate families](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/leaderboard.png)

The [complete rankings and per-family limitations](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/complete_rankings.md) cover all 36 completed variants. [Best observed calibrated-F1 variants per machine and scored span](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/best_per_machine.csv) are also provided, explicitly as retrospective selections.

## 3. Stability: what persists during training?

I separate three questions:

1. **Does the normal/injected validation objective fluctuate?** This measures the proxy training task and validation noise.
2. **Do score thresholds and calibrated decisions fluctuate?** This is closer to the practical alarm problem.
3. **Does a good state occupy a broad training plateau?** This measures how easy it might be to find a state without guessing one lucky epoch.

“Late” means the final third of each recorded training run; all completed variants have 300 validation epochs and 30 scheduled test evaluations, every 10 epochs. For decision/threshold summaries, recorded epochs 200–300 are included. F1 SD is computed within each machine and then averaged. Threshold CV is within-run standard deviation divided by the absolute mean. Thus 1.2% CV is steadier than 10%, but it is not a bound on the next threshold jump. “Near” means within **0.02 absolute F1**, or two F1 percentage points; it is an explicit analysis tolerance, not a tuning result.

| Experiment | Late F1 SD (points) | Late threshold CV | Near own best calibrated F1 (all recorded evals) | Selected calibrated F1 within 0.02 of oracle |
| --- | --- | --- | --- | --- |
| frequency | 6.22 | 8.76% | 23.0% | 4/28 |
| frequency_aux | 5.72 | 10.41% | 13.7% | 2/28 |
| frequency_aux_no_film | 5.25 | 11.65% | 11.4% | 2/28 |
| ae/default | 0.72 | 2.23% | 60.0% | 3/28 |
| vae/default | 0.58 | 2.76% | 45.6% | 3/28 |
| frequency / recon / default | 0.82 | 1.72% | 49.5% | 2/28 |
| frequency_aux / recon / auxilary_added | 0.39 | 1.20% | 65.7% | 2/28 |

The frequency_aux reconstruction arm has late calibrated-F1 SD **0.39 points** and threshold CV **1.20%**, compared with **6.22 points / 8.76%** for frequency phase one. AE is **0.72 points / 2.23%**. The stable reconstruction arm occupies its own near-best logged calibrated plateau on **65.7%** of scheduled checkpoints; frequency does so on **23.0%**. However, the reconstruction arm is within two points of the same checkpoint's oracle on only **8.6%** of scheduled evaluations. It is often **stably suboptimal**.

At the selected checkpoints, perturbing the calibrated threshold by ±10% gives a mean worst-side F1 drop of **0.0394** for that reconstruction arm, **0.0517** for AE, and **0.0576** for frequency. This is a retrospective sensitivity test, not observed deployment drift. Individual losses can be substantially larger. A threshold that barely moves can still lie on a steep part of the decision curve.

![Training stability and calibration gap measure separate properties](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/stability.png)

### An important early-checkpoint pattern

The minimum TensorBoard validation loss occurs at epoch **1 or 2 on all 28 frequency_aux runs**; 22 select epoch 1. Frequency without aux selects epoch 1 on 18/28 machines. Several selected early phase-one checkpoints have better oracle scores than every checkpoint in the evaluation history, which begins at epoch 10. Consequently, “best logged epoch” is not the absolute best trained state, and the final-epoch curve is not the selected detector.

This does not prove early weights understand anomalies better for a causal reason. It establishes that longer training does not consistently improve the selection objective, and that the monitoring cadence misses the selection region. Log epochs 1, 2 and 5 in future runs before deciding that a 300-epoch run has found a useful plateau.

Validation loss is a weak universal checkpoint proxy for calibrated F1: the mean within-run Spearman correlation is **−0.023 for frequency**, **−0.193 for AE**, and **−0.245 for the stable frequency_aux reconstruction arm**. A perfect inverse relationship would be −1. These correlations are descriptive, affected by temporal trends and changing corruptions; they are not significance tests.

Fresh synthetic corruptions are drawn during LEWM validation, so validation jitter includes **changing task difficulty**, not just changing model weights. A fixed validation corruption bank is needed to isolate weight stability. Logged aggregate phase-one latent variance remains nonzero, with family median values around 1.1 at the first logged batch and 1.9–2.1 at the last. These scalars do **not** support a claim of global representation collapse, and cannot rule out loss of particular fault-sensitive directions.

## 4. Why the clean threshold frequently misses the oracle

Each machine uses approximately **2,369–2,875 nominal timesteps** in its held-out train tail. With input length 1,024 and stride 10, calibration has only **136–187 window scores**. Adjacent windows share 1,014 of 1,024 samples. At quantile 0.995, NumPy's linear interpolation falls **between the largest two scores** for all these calibration sample sizes. The tail spans only about 2.3–2.8 input-window widths; the raw count of overlapping windows is not an independent sample-size estimate.

This makes the threshold sensitive to a small nominal episode and to which nominal regime happens to occupy the tail. It also explains why smooth average validation loss need not imply a well-estimated extreme score quantile. The same tail is used for checkpoint validation and threshold calibration. The scaler uses the full nominal training file, including the tail: there is no test normalization leakage, but the preprocessing holdout is not fully independent.

### An oracle quantile of 0 or 1 is often a warning about reachability

The saved diagnostic clips the oracle's equivalent clean quantile to [0,1]. If the oracle threshold exceeds the largest clean score, `q=1` reproduces the clean maximum, **not the oracle threshold**. A repeated q=1 can therefore hide a moving, unreachable optimum. The analogous problem occurs below the clean minimum.

| Experiment | Exact oracle threshold inside clean range | Within 0.02 of oracle provably impossible using clean empirical quantiles |
| --- | --- | --- |
| frequency | 7/28 | 18/28 |
| frequency_aux | 7/28 | 18/28 |
| frequency_aux_no_film | 5/28 | 20/28 |
| ae/default | 6/28 | 17/28 |
| vae/default | 7/28 | 18/28 |
| frequency / recon / default | 7/28 | 20/28 |
| frequency_aux / recon / auxilary_added | 7/28 | 18/28 |

The final column is stronger than the exact-threshold check. For out-of-range optima, I scanned test F1 under a **relaxed, one-sided clean-range restriction**, allowing even thresholds beyond the unknown opposite clean endpoint. If that generous upper bound still falls below oracle−0.02, no empirical clean quantile can get within two points. Thus **18/28 frequency runs** provably fail this near-oracle criterion at their selected checkpoint. The other cases are not proven reachable: an out-of-range oracle might have an equally good in-range plateau, and passing this upper-bound check cannot establish that.

### Three concrete failure cases

- **Frequency / machine-3-11:** calibrated F1 **0.1953**, oracle **0.9686**, VUS-PR **0.9629**, VUS-ROC **0.9910**. The calibrated threshold is **0.0810**, versus oracle **10.6667**. Test normal-window median score is **0.0985**, already above the clean threshold; **63.1% of normal test windows alert**. Recall is 100%. This is strong discrimination with severe threshold transfer failure, not evidence that the model cannot rank faults.
- **Frequency / machine-1-7:** calibrated F1 **0.2850**, oracle **0.7797**. The clean threshold **0.2452** is too high for much of the test distribution; recall is **16.6%** and normal-window FPR is 0%. The oracle threshold **0.0701** lies below the minimum clean score and would alert on every calibration window. The same clean-tail rule cannot simultaneously explain normal calibration and optimal test behavior.
- **Frequency / machine-3-3:** oracle F1 **0.9397** exactly equals the F1 of **always predicting anomalous**. Here **88.6%** of scored windows are positive. VUS-PR **0.9296** accompanies VUS-ROC **0.5768**; the large PR/F1 values are not sufficient evidence of useful discrimination. Calibrated F1 is **0.2887**.

![F1 versus threshold for three different failure cases](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/threshold_examples.png)

Across the full-span task, the always-alert classifier has mean machine F1 **0.5565**, higher than every completed variant's current calibrated mean. That classifier is operationally undesirable, and this comparison does not erase the detectors' lower false-positive rates or useful rankings. It shows why **F1 must be read alongside prevalence, recall and normal-window false positives**. For example, frequency's macro normal-window FPR is **42.1%** despite its leading calibrated mean F1. These are overlapping-window rates, not independent alert counts or false alarms per hour.

![Validation, F1 and threshold histories for representative frequency failures](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/training_examples.png)

The selected early-checkpoint numbers above and these scheduled training trajectories refer to different checkpoints. In particular, the plot's first historical point is epoch 10; it does not reproduce a selected epoch-1 metric.

## 5. Why VUS-PR, VUS-ROC and F1 can disagree

They summarize different operational tradeoffs. Ordinary ROC relates true-positive rate to the fraction of normal windows falsely alerted. PR relates recall to the fraction of alarms that are correct. At prevalence π,

`precision = π × TPR / (π × TPR + (1 − π) × FPR)`.

Thus ROC can reward broad ranking improvements while PR rewards a much cleaner high-score tail. If curves cross, one area can increase while the other decreases. Class imbalance does not automatically inflate ROC when conditional score distributions are fixed; instead, **global ROC area can conceal the low-FPR behavior you actually need**, and PR changes with prevalence. These distinctions are demonstrated in the primary study by [Saito and Rehmsmeier](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432).

### A verified reversal within this repo

Both rows below use **machine-2-3, input 1,024, target output 512, stride 10**, with identical saved window labels and prevalence **20.24%**.

| Observed-query encoder | VUS-PR | VUS-ROC | Calibrated F1 | Oracle F1 | Recall at ≤1% ordinary FPR |
| --- | --- | --- | --- | --- | --- |
| time_aux | 0.6110 | 0.7246 | 0.3851 | 0.6771 | 0.5185 |
| frequency_aux | 0.5057 | 0.8005 | 0.5509 | 0.5849 | 0.2331 |

The time_aux model gives **100% precision among its highest-scoring 5% of windows**, versus **89.5%** for frequency_aux. At an ordinary 1% false-positive rate it recovers **51.9% versus 23.3%** of positives. The time model has the more useful early retrieval region and better PR. The frequency model's broader ranking earns higher total ROC. Their ordinary ROC areas are 0.8253 and 0.8387, while ordinary average precisions are 0.7404 and 0.5829; neither ordinary area is being mislabeled as VUS.

Despite its stronger PR and oracle F1, time_aux has **worse calibrated F1**, because its chosen clean threshold lands at a worse operating point. This example contains all three effects: crossing ranking curves, different area emphasis, and calibration error.

![PR and ROC curves for a controlled same-machine reversal](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/metric_divergence.png)

VUS adds range-aware label buffers and averages areas over a finite range of tolerances; it also rewards covering anomaly ranges. It therefore differs further from one-threshold binary F1. The [original VUS paper](https://www.vldb.org/pvldb/vol15/p2774-paparrizos.pdf) and [authors' implementation](https://github.com/thedatumorg/VUS) define this range treatment. Full details and primary-source links are in the [metric semantics appendix](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/metric_semantics.md).

Within a machine and fixed scored span, VUS-PR and VUS-ROC usually agree broadly: the mean family-rank Spearman correlation is **0.899 for full span** and **0.862 for half span**. Their occasional reversals are meaningful exceptions. Calibrated F1 correlates much less with VUS-PR across methods: **0.268 / 0.362**, respectively. Calibration often breaks the connection between good ranking and good decisions.

## 6. Window size itself changes the apparent winner

The saved inference sweep contains **162 experiment/case combinations, each on all 28 machines**. It reuses selected weights and recalibrates under each geometry; it is not new training.

For frequency phase one:

| Input and scored length | Calibrated F1 | Oracle F1 | VUS-PR | VUS-ROC |
| --- | --- | --- | --- | --- |
| 1024 | 0.5201 | 0.7490 | 0.7039 | 0.7028 |
| 512 | 0.4032 | 0.6400 | 0.5674 | 0.7105 |
| 256 | 0.3619 | 0.5599 | 0.4727 | 0.7340 |
| 128 | 0.3275 | 0.5144 | 0.4187 | 0.7577 |

Shorter windows improve ROC while decreasing PR and F1. Changing the scoring span changes which windows count as anomalous, the anomaly fraction within them, the averaging operation, coverage, and the normal calibration distribution. This is not a controlled claim that one representation became simultaneously better and worse. The same checkpoint is solving a different window task.

In the smaller-input sweep, the highest calibrated mean is **frequency_aux_no_film at input 512: 0.4171**. That does not overturn the base 1,024-window ranking. Among cross-attention cases, averaged over completed heads, keeping target length 128 but expanding context from 256 to 512 gives calibrated F1 **0.2862 → 0.2829**, with nearly unchanged VUS-PR/ROC. More context did not produce a general improvement in those saved cases.

The positional variants have only **5 machines (features)** and **6 machines (query)**, and use **input 768 / target 256**, unlike the completed 1,024 / 512 variants. Their apparent mean ROC values of **0.8178 / 0.7815** cannot establish a positional-encoding win. Complete the machine set and compare the same input/target geometry and initial weights, changing only position encoding.

## 7. Why the model can fail the actual anomaly task

These are source-verified mechanisms and measured symptoms. They are not all established causes of the observed errors.

**Normal restoration is a proxy for anomaly separation.** AE learns normal reconstruction, VAE adds posterior KL, phase-one LEWM restores injected examples, and phase-two heads learn another restoration target. Low normal prediction error helps only if real anomalous windows retain a distinctly larger error. Flexible restoration can also explain a fault; benign regime changes can be more surprising than a repeatable fault.

**Phase-one training and scoring have different geometry.** The saved phase-one configs use symmetric channel-softmax KL, but scoring uses raw latent L1. Adding a common channel offset to a prediction vector independently of its target leaves channel softmax unchanged and can change raw L1 substantially. This is an exact mismatch, so better validation KL need not imply a stable L1 threshold. Match the loss and score in a controlled comparison rather than assuming the best loss gives the best detector. AE/VAE train MSE while using L1 scores; these also weight large residuals differently, and their relative practical impact is unmeasured.

**Training masks and inference actions differ.** Phase-one training uses the known synthetic injection mask. Scoring uses an auxiliary-proposed soft mask, or no mask in a no-aux arm. The auxiliary is trained on artificial corruptions, whose match to real SMD fault mechanisms has not been established. Failure could reflect bad proposals, wrong restoration, or insufficient fault evidence; aggregate F1 cannot distinguish these.

**Observed cross attention sees the target region.** It is conditional reconstruction, not an unseen-future forecast. Although attention has no direct Query residual shortcut, target observations control retrieval. In query-target variants, the clean target Query representation is itself trainable without detach. A model can reduce a shared representation loss without preserving every fault-sensitive distinction. SIGReg discourages trivial collapse but does not guarantee useful anomaly semantics.

**Mean scores can dilute sparse evidence, while any-positive labels demand detection.** I audited 112 archives across frequency, frequency_aux, AE and frequency_aux observed_query. For machines containing both occupancy strata, recall for windows with ≤10% anomalous timesteps versus >50% was **0.683 vs 0.811** for frequency, **0.663 vs 0.818** for frequency_aux, and **0.779 vs 0.909** for AE (11 eligible machines each). The query arm gives **0.551 vs 0.730** across 16 eligible machines. This supports a difficulty with short-overlap/boundary windows. It does not prove mean pooling caused the misses: fault type, amplitude and thresholds also differ. A fixed-checkpoint pooling comparison can test that hypothesis.

**The encoder may retain the wrong distinctions.** A frozen downstream head cannot recover information already discarded by the encoder. Nonzero global latent variance does not prove normal/fault separation. Conversely the strong frequency rankings on machine-3-11 show that some apparent model failures are threshold failures rather than absent fault information.

The [model design appendix](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/model_design.md) links every mechanism to code/config lines, documents the occupancy audit and lists falsifiable follow-up experiments.

## 8. Evaluation shortcomings that must be fixed before trusting small gains

1. **Mixed summary columns.** Existing training summaries place window F1 beside unqualified VUS values from another protocol. This report bypasses them and reads explicit window metrics. Future summary columns should name their protocol.
2. **Current VUS source and saved artifacts disagree.** The checked-out wrapper doubles its buffer despite a direct-buffer docstring/spec. Three sampled saved runs instead reproduce the single-buffer convention within 1.4×10⁻⁸. Applying today's wrapper changes their VUS-PR by **+0.0020 to +0.0134** and ROC by **+0.0025 to +0.0127**. This report retains saved values. Three samples do not certify every artifact; publish the metric convention and implementation revision, and standardize it before judging tiny differences. The metric code was not modified in this audit.
3. **“Best” checkpoint names are ambiguous.** Some LEWM best-F1 weights were selected using an excluded test protocol. Main rankings here use metadata identifying validation-selected weights. AE/VAE test-best checkpoints and any test-oracle epoch are upper-bound diagnostics, not label-free selections.
4. **Validation/calibration are small and reused.** Use separate nominal blocks where possible. The present score archive does not save the full clean calibration distribution, preventing a complete retrospective clean-quantile plateau or calibration-bootstrap analysis without rescoring.
5. **A single seed is not reproducibility.** Epoch stability at seed 4 does not establish restart stability. Machine-to-machine SD is not seed SD. Several no-Film reconstruction `default` and `auxilary_added` rows have identical saved metric profiles on every machine; they are not independent evidence of improvement.
6. **The recorded threshold is recalibrated each evaluation.** Stable recalibrated F1 does not show that one threshold frozen at epoch 20 would work at epoch 300. Saved historical model/score pairs would be needed for that test. Sparse test evaluations also miss excursions between epochs.
7. **Window F1 is not an alert policy.** Overlap produces correlated repeated decisions. Useful deployment evaluation needs grouped alarm episodes, delay, and a false-alarm budget on genuinely nominal future blocks. The sampling cadence and operational costs were not supplied, so no false alarms/hour or time-to-detect claims are invented here.
8. **This comparison uses the test set repeatedly.** Architecture shortlists and the explanations in this report are exploratory. Freeze the next selection/calibration policy and evaluate it on an untouched labeled set or predeclared held-out blocks before making a final generalization claim.

## 9. Concrete next steps, in order

| Priority | Action | What it resolves |
| --- | --- | --- |
| 1 | Freeze and version window labeling, score span, stride, metric buffer and checkpoint provenance; name protocol in every CSV column | Prevents apparent gains caused by different tasks or stale metric code |
| 2 | Freeze AE and the stable frequency_aux reconstruction checkpoint; calibrate separately on several disjoint nominal blocks, saving the clean scores | Tests whether threshold transfer, rather than architecture, is the dominant bottleneck |
| 3 | Choose a practical nominal false-alarm target, report window precision/recall and grouped alerts, and include always-alert F1 | Makes “useful” measurable without rewarding trivial high-prevalence solutions |
| 4 | Use a fixed corruption bank for checkpoint validation; log epochs 1, 2, 5, 10 onward and retain early selected weights | Separates validation randomness from model instability and captures the observed selection region |
| 5 | Test matched training/scoring geometry for phase one: KL/KL versus KL/L1 versus L1/L1 under the same seed/data | Tests the concrete objective mismatch rather than guessing why validation and detection disagree |
| 6 | At fixed weights and a fixed labeling span, compare mean with a predeclared top-k/channel-aware reduction and inspect low-occupancy misses | Tests whether averaging hides short/sparse faults; recalibrate each score definition independently |
| 7 | For cross attention, compare input/features/query targets and position encoding with identical geometry and machines | Tests moving targets and temporal addressing without mixing task/prevalence changes |
| 8 | Repeat the shortlisted arms across seeds and nominal calibration intervals, then audit on held-out labeled data | Establishes whether the apparent gains and stable thresholds survive restart and regime variation |

The immediate research question should be **“Can a frozen, reasonably stable scorer meet a declared false-alarm budget across different nominal regimes?”** If the answer is no, another small VUS improvement will not by itself produce a useful detector. If yes, the frequency ranking gains become much more actionable.

## 10. Files, verification and reproduction

Main data: [family summary](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/family_summary.csv), [per-machine selected runs](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/selected_runs.csv), [window metrics by epoch](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/epoch_metrics.csv), [inference-case means](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/inference_case_summary.csv), [threshold reachability bounds](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/threshold_reachability.csv), [occupancy diagnostics](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/window_occupancy.csv), [nominal calibration sizes](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/calibration_sizes.csv), [compressed TensorBoard scalar export](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/training_scalars.csv.gz). The [per-machine heatmap](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/machine_heatmap.png) makes failures hidden by averages visible.

All 1,019 stored calibrated F1 values were reproduced **exactly** from saved window scores and labels. Oracle F1 agreed within **2.23×10⁻¹⁶**. Recorded scored-span lengths match the saved intervals; all complete families cover the same 28 machines; no duplicate history epochs or TensorBoard scalar steps were found in the extracted streams. The separate occupancy audit also verified label and strict-threshold prediction contracts on 112 archives. Three VUS reproductions and the buffer spy are documented in [vus_artifact_verification.json](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/vus_artifact_verification.json) and the metric appendix.

The scripts write only inside this report directory and read existing results/datasets. No training, inference rescore of models, model-code edit, or experiment-artifact mutation was performed. Computational metric evaluation from saved scores is included in verification. Existing uncommitted workspace changes were preserved.

```bash
./venv/bin/python reports/smd_window_analysis/analyze.py
./venv/bin/python reports/smd_window_analysis/audit_thresholds.py
./venv/bin/python reports/smd_window_analysis/audit_window_occupancy.py
./venv/bin/python reports/smd_window_analysis/metric_semantics_probe.py
./venv/bin/python reports/smd_window_analysis/verify_vus_artifacts.py
./venv/bin/python reports/smd_window_analysis/summarize.py
./venv/bin/python reports/smd_window_analysis/write_report.py
```

CSV exports follow this repo's existing `*.csv` ignore rule and remain available locally. The raw training-scalar export includes TensorBoard steps: `Train Loss/latent_var` uses batch steps, whereas `train/*` and `validation/*` use epoch steps. Only epoch-indexed validation tags enter the validation analysis. Source hashes and metric-file hashes are saved in [provenance.json](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_window_analysis/provenance.json); inspected source HEAD was `31427a6b9ae09524f9ac53fab3e930cf1823262e` with pre-existing working-tree modifications. The appendices audit that working tree, which is not assumed to be the implementation that produced every historical run.
