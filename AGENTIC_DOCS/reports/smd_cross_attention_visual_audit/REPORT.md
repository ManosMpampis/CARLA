# Are the frequency_aux observed-features scores useful?

30 October 2026 · SMD · all 28 machines · `lewm_encoder/frequency_aux/cross_attention/observed_features` · selected validation-loss weights. No retraining or fresh model inference.

**Yes: your observation is supported by the actual traces. This head contains useful anomaly information on many machines. Its low average calibrated F1 does not establish that cross attention has failed.** The largest final replay score lies inside a labeled anomaly on **26/28 machines**. Among the highest-scoring 1% of covered timesteps, mean per-machine precision is **58.7%**, compared with mean anomaly prevalence **4.29%**. That is a substantial concentration of fault evidence.

The qualified conclusion is that **some machines mainly have a calibration problem; others also have an alarm-duration/localization problem; a smaller set has genuinely weak score separation**. One threshold change cannot solve all three. Ten machines have window-oracle F1 ≥0.70, while four remain ≤0.50 even with a test-chosen window threshold. The pointwise ceiling is different: six machines reach ≥0.70 and seven remain ≤0.35.

I loaded **every `decision_trace.json`**, reconstructed the final timeline and the original window contributions, and saved a graph for every machine. Browser policy blocks local `file:` pages, so these are **renderings of the actual JSON graph values, not screenshots of the HTML UI**. Three original HTML embedded payloads were independently checked against their JSON. The [offline image gallery](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/GALLERY.html) contains all 28 images; the machine table below links individual figures.

## 1. What the graph and window metric measure

| View | Score | Label | Threshold |
| --- | --- | --- | --- |
| HTML/full-series replay | Mean of all overlapping contributions to a timestep | Actual anomaly label at that timestep | Timestep clean-calibrated threshold |
| Window evaluation | Mean error across one 512-timestep scored target | Positive if **any** anomaly occurs anywhere in those 512 timesteps | Separate window clean-calibrated threshold |

A single obvious fault peak may be exactly where you expect and still have a poor score for different reasons:

- Its response extends outside the labeled event, producing false-positive duration.
- It catches the onset but stays low during the rest of the fault, producing missed duration.
- Mean pooling dilutes a short peak across 512 positions, while the any-positive window label covers many windows around that short event.
- It detects one visually dominant event but misses many smaller ones.
- Normal score levels change across the series, so one global threshold cannot separate every regime.

The first 512 timesteps provide context and receive no scores. Across the 28 recordings, five labeled events lie entirely in that startup region; **all event denominators here exclude those five unscored events**, while the pictures still show their truth bands. There are 322 covered events and 327 total events. Timestep metrics likewise use covered positions. This coverage restriction is shared by the matched heads.

These are measurable effects here, not reasons to disregard the visual evidence. Also, a better threshold changes F1 but **does not change VUS** for an unchanged score vector: VUS already integrates threshold-based ranking behavior. A window VUS value must be interpreted using the window score/label definition, not inferred from the different timestep curve.

In the pictures, blue is the actual score, green dashed is calibration, and orange dotted is the separately optimized test oracle. Scores are divided by their own calibrated threshold, so the current threshold is always **1**. The vertical transform is the HTML's nonnegative `log1p(score/threshold)` transform. The final curve gets a fitted vertical range instead of the HTML's full historical range; the pixels and UI are not identical. No smoothing, clipping, or label-based score transformation is applied. Purple denotes actual timestep faults in the top panel and **positive-window center ranges** in the middle panel. Window centers are display positions, not causal alarm times.

## 2. Your strongest example: useful signal hidden by calibration

### Machine-2-8

The highest response aligns with the only labeled event, and the separation is excellent:

| Diagnostic | Calibrated | Oracle on the same saved scores |
| --- | ---: | ---: |
| Window F1 | 0.0919 | **0.9771** |
| Timestep F1 | 0.0296 | **0.9369** |
| Normal timestep false-positive rate | 45.8% | 0.07% |

The event lasts **161 timesteps**. At calibration the peak-containing continuous alarm spans **6,207 recorded timesteps and continues to the recording's end**, including the elevated post-event baseline. At the point-F1 oracle it lasts **172 timesteps**, with precision **90.7%**. Ordinary timestep AP is **0.9899**, ROC **0.9999**; window VUS-PR/ROC are **0.9411/0.9820**. This is direct evidence that the representation can distinguish the event and that the supplied threshold makes it look much worse than its ranking warrants.

![Machine-2-8: excellent signal, poor calibrated threshold](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-2-8.png)

The oracle is a retrospective ceiling chosen with test labels, not a threshold the existing clean-tail method knows how to find. Nevertheless, this case strongly supports keeping the architecture and investigating calibration.

### Machine-3-11

Window F1 improves **0.0947 → 0.9174**, with window VUS-PR/ROC **0.9335/0.9832**. All three labeled events are hit at calibration, but **61.8%** of normal timesteps also alert. There is clear window-level potential. The best timestep F1 is only **0.4961**, so improving the threshold does not completely solve duration localization. At the final maximum, a 6-timestep labeled event sits inside a 146-timestep alarm even at the point-F1 oracle; nearby labeled events can also contribute to the alarm's precision.

### Machine-3-6

This is useful even with the current threshold: window F1 **0.7759**, timestep F1 **0.6153**, and **11/11 events hit**, with normal timestep FPR **4.6%**. The corresponding oracle ceilings are **0.8705 / 0.7365**. It is a good reference case for inspecting what this head learns successfully.

## 3. Why a convincing peak can still yield low F1

### The peak is right, but its alarm is too broad

On **machine-2-3**, the maximum lies in a **6-timestep** labeled event. The calibrated alarm around it spans **110 timesteps**; the oracle alarm still spans **94**. At calibration the head hits **9/10 events**, yet timestep F1 is only **0.1069**. Detecting the event and localizing its duration are different successes. Window F1 is much higher, **0.5248**, but already close to its **0.5319** oracle ceiling.

On **machine-1-4**, the maximum is also a true anomaly. Its containing labeled event lasts **46 timesteps**, while the calibrated peak alarm lasts **773**. Earlier faults are not consistently separated either. Window F1 is **0.5446 → 0.5458**: here calibration is already near-optimal for the existing window score, so a different window threshold cannot deliver a dramatic gain.

Coarse encoded features, receptive fields and overlapping reconstruction can plausibly broaden responses. This audit establishes the width mismatch but does not isolate which mechanism caused it. Very long post-event plateaus can exceed the whole 1,024-timestep input length; overlap alone cannot explain every such plateau in a stateless window scorer. Those cases need inspection of the nominal telemetry regime and model error after the event.

### One large fault can dominate point F1 while many events are missed

On **machine-1-6**, timestep F1 improves **0.3176 → 0.8840**. But the point-F1 oracle hits only **4/29 events**, versus **28/29** at calibration: it removes many small-event alarms while concentrating on the long late fault. On **machine-2-7**, timestep oracle F1 is **0.8224**, yet only **2/20 events** are hit. These can be useful detectors of dominant faults, but those F1 values do not mean broad fault coverage.

The event-hit counts here are the simple fraction of covered labeled intervals containing at least one positive prediction. They are a diagnostic, **not** point-adjusted F1 and not a full event matching metric. A high event hit rate can still come with many false alerts.

### Sometimes the scores themselves are weak

On **machine-3-2** and **machine-3-3**, the maximum score is outside every labeled fault. Timestep oracle F1 is **0.1349 / 0.0744**, and recall at ≤1% ordinary timestep FPR is only **3.0% / 3.0%**. These are not rescued by choosing a better global threshold.

Machine-3-3's window oracle F1 is **0.7457**, but an always-alert window classifier already obtains **0.7071** because 54.7% of scored windows contain some anomaly. That window F1 should not overrule the weak timestep separation visible in the graph. It is a different, high-prevalence labeling task.

![Machine-3-3: genuinely weak timestep separation and a high-prevalence window task](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-3.png)

## 4. What happens across all machines

All means below give each machine equal weight. Timeline statistics come from the JSON final states; window statistics use the official NPZ window scores.

| Quantity | Mean across 28 machines |
| --- | ---: |
| Window calibrated / oracle F1 | **0.4046 / 0.6489** |
| Timestep calibrated / oracle F1 | **0.2308 / 0.5022** |
| Ordinary timestep AP / ROC | **0.4632 / 0.8811** |
| Event hit fraction, calibrated / timestep-oracle threshold | **70.3% / 56.7%** |
| Normal timestep FPR, calibrated / timestep-oracle threshold | **21.2% / 5.18%** |
| Timestep recall at ≤1% ordinary FPR | **40.4%** |

This is useful ranking information, but the default threshold has substantial false-alarm burden. The oracle raises ordinary point F1 on average while reducing equal-weight event coverage. A lower-FP operational threshold and an event-sensitive objective may therefore favor different settings. The 1% FPR diagnostic uses test labels to characterize the curve, not a deployable threshold selection rule.

### Is observed-features actually worse than other cross-attention heads?

The matched comparison uses identical machines, input length 1,024, target length 512, stride 10, labels and scored support. These are the right controls for judging this architectural head.

| Variant | Window calibrated F1 | Window oracle F1 | Window VUS-PR | Window VUS-ROC |
| --- | ---: | ---: | ---: | ---: |
| frequency_aux observed_features | 0.4046 | **0.6489** | 0.5655 | 0.7024 |
| frequency_aux observed_query | 0.4150 | 0.6404 | 0.5692 | 0.7059 |
| frequency_aux observed_input | 0.3450 | 0.5852 | 0.4980 | 0.6512 |
| frequency_aux_no_film observed_features | 0.4060 | 0.6564 | 0.5647 | 0.6984 |

**Observed-features is competitive with Query and no-FiLM Features, and clearly stronger than observed Input on most machines.** It beats observed Input on calibrated window F1 on **20/28** machines and on oracle F1 on **24/28**. Against Query, calibrated wins split **14/14**, while oracle F1 gives Features **17 wins, 10 losses, 1 tie**. Its matched NPZ timestep AP/ROC are **0.4632/0.8811**, versus Query's **0.4532/0.8666**. These are descriptive comparisons at seed 4, not proof of a statistically established global winner.

There are genuine counterexamples: Query is better on machine-3-7, and no-FiLM Features is better on machine-1-2. The [matched-head comparison](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/matched_heads.md) details those cases. The earlier full-span-versus-target-half headline averages cannot establish that cross attention is architecturally inferior, because those methods have different scored regions and window labels.

## 5. Machine-by-machine visual judgment

Every row links a full-resolution picture: timeline, window means, and both threshold/F1 curves. “Oracle” means the best ordinary F1 on that particular saved test score vector; the timeline and window oracles are separate.

| Machine / full image | Window F1 calibrated → oracle | Timeline F1 calibrated → oracle | Covered events hit at calibration | Normal timestep FPR | Visual interpretation |
| --- | --- | --- | --- | --- | --- |
| [1-1](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-1-1.png) | 0.534 → 0.730 | 0.431 → 0.608 | 8/8 | 27.0% | The central faults have clear peaks. Normal shoulders and background excursions also exceed calibration; raising the threshold helps, while small late events remain easy to overlook. |
| [1-2](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-1-2.png) | 0.579 → 0.585 | 0.384 → 0.521 | 4/10 | 0.5% | Some events are distinct and others weak. Window calibration is already close to its oracle; a different window threshold alone cannot produce a large gain. |
| [1-3](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-1-3.png) | 0.221 → 0.550 | 0.218 → 0.352 | 4/11 | 3.9% | The largest late peak is a real fault, but its calibrated alarm lasts much longer than that event. Earlier events are not consistently separated from background. |
| [1-4](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-1-4.png) | 0.545 → 0.546 | 0.255 → 0.280 | 9/11 | 8.9% | The late peak is genuine, yet the alarm extends far beyond it and several earlier faults are weak. Window F1 is already near its threshold ceiling. |
| [1-5](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-1-5.png) | 0.494 → 0.725 | 0.117 → 0.514 | 6/7 | 6.2% | Short labeled events produce sharp high peaks. A higher threshold removes substantial background activity, but the remaining localization is not perfect. |
| [1-6](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-1-6.png) | 0.789 → 0.823 | 0.318 → 0.884 | 28/29 | 81.6% | The long late fault is strongly separated after raising the timeline threshold. However, the point-F1 oracle then hits only 4 of 29 events, sacrificing many short faults. |
| [1-7](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-1-7.png) | 0.264 → 0.650 | 0.090 → 0.453 | 4/13 | 0.6% | Many positive windows lie below calibration, so a lower window threshold helps. The timeline still has substantial score overlap and only moderate threshold potential. |
| [1-8](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-1-8.png) | 0.629 → 0.630 | 0.167 → 0.324 | 5/20 | 6.1% | Visible peaks do not cover many of the labeled events. Window calibration is nearly optimal already; better score separation is needed for a major gain. |
| [2-1](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-2-1.png) | 0.652 → 0.657 | 0.313 → 0.336 | 10/13 | 4.2% | Some peaks coincide with faults, but similar excursions occur elsewhere and event interiors can be quiet. Both calibrated scores are close to their oracle ceilings. |
| [2-2](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-2-2.png) | 0.695 → 0.791 | 0.466 → 0.470 | 11/11 | 13.3% | High onset responses are visible, yet many labeled interiors have lower scores. Timeline threshold tuning barely improves F1; window averaging has more headroom. |
| [2-3](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-2-3.png) | 0.525 → 0.532 | 0.107 → 0.266 | 9/10 | 17.6% | A high peak is real, but the peak-containing labeled event lasts only 6 timesteps while the calibrated alarm lasts 110. Event hits can look good while duration precision is poor. |
| [2-4](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-2-4.png) | 0.316 → 0.654 | 0.318 → 0.368 | 2/20 | 1.7% | One dominant long event is reconstructed well; most other events are missed at calibration. Lowering the window threshold helps, but the point-F1 optimum focuses on long events. |
| [2-5](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-2-5.png) | 0.321 → 0.731 | 0.300 → 0.356 | 18/21 | 4.1% | Several positive windows sit just below calibration. Lowering the window threshold improves F1 substantially, whereas precise timestep localization remains harder. |
| [2-6](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-2-6.png) | 0.601 → 0.630 | 0.169 → 0.453 | 7/8 | 5.9% | Visible peaks contain real anomalies. Calibrated alarms have long tails; a higher timeline threshold narrows them, while the window result is already moderately useful. |
| [2-7](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-2-7.png) | 0.290 → 0.622 | 0.693 → 0.822 | 3/20 | 0.8% | The dominant event is strongly detectable, but only 3 of 20 events are hit at calibration and 2 at the point-F1 oracle. High point F1 is not broad event coverage. |
| [2-8](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-2-8.png) | 0.092 → 0.977 | 0.030 → 0.937 | 1/1 | 45.8% | A particularly strong calibration-failure case: almost perfect ranking and a clear real peak, but a post-event score plateau remains above the calibrated threshold. |
| [2-9](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-2-9.png) | 0.548 → 0.792 | 0.472 → 0.758 | 10/10 | 14.3% | Several central faults are strongly visible. A higher threshold suppresses surrounding normal responses; the detector retains meaningful event and window information. |
| [3-1](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-1.png) | 0.179 → 0.589 | 0.042 → 0.578 | 4/4 | 50.7% | Real early/late faults are visible, but the calibrated threshold also admits many normal excursions. Threshold repair helps; the window ceiling remains moderate. |
| [3-10](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-10.png) | 0.310 → 0.464 | 0.381 → 0.655 | 3/12 | 10.9% | Major faults are visible, especially the late one. Periodic window-score excursions also occur outside positive windows; timestep and window usefulness differ sharply. |
| [3-11](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-11.png) | 0.095 → 0.917 | 0.022 → 0.496 | 3/3 | 61.8% | The rare, very large fault responses support strong window detection potential. Calibration floods normal regions, and the timestep responses are much broader than short fault labels. |
| [3-2](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-2.png) | 0.075 → 0.570 | 0.000 → 0.135 | 0/9 | 0.0% | The highest score is normal and calibration raises no alarms. Lower thresholds recover some positives at a large false-positive cost; precise localization remains weak. |
| [3-3](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-3.png) | 0.088 → 0.746 | 0.037 → 0.074 | 5/26 | 2.7% | The highest score is normal and labeled event scores overlap background. Oracle window F1 is only modestly above always-alert F1; precise timestep discrimination is very weak. |
| [3-4](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-4.png) | 0.362 → 0.492 | 0.089 → 0.798 | 8/8 | 90.2% | A clear first-event peak is followed by a long elevated normal-score plateau. A higher threshold detects the major fault well, but its point-F1 oracle hits only 2 of 8 events. |
| [3-5](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-5.png) | 0.464 → 0.507 | 0.106 → 0.540 | 7/11 | 25.0% | A real peak is followed by a long calibrated alarm outside most labels. Raising the threshold helps point F1, but misses many smaller events. |
| [3-6](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-6.png) | 0.776 → 0.870 | 0.615 → 0.736 | 11/11 | 4.6% | One of the clearest already-useful cases: multiple fault responses, good current window F1, and all 11 events hit at calibration. Some threshold headroom remains. |
| [3-7](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-7.png) | 0.275 → 0.325 | 0.086 → 0.485 | 5/5 | 31.6% | A real onset peak is followed by a persistent elevated background. Better calibration helps the timeline, but window F1 has a low ceiling; Query genuinely performs better here. |
| [3-8](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-8.png) | 0.433 → 0.500 | 0.199 → 0.303 | 6/6 | 21.8% | The normal-score background rises and many peaks occur. One global threshold cannot cleanly separate all events from that changing background. |
| [3-9](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machines/machine-3-9.png) | 0.178 → 0.565 | 0.038 → 0.559 | 4/4 | 52.4% | Real fault peaks coexist with many elevated normal responses, especially near the end. Calibration is too permissive; threshold tuning helps but does not make the window task easy. |

## 6. All-machine contact sheets

These cover every machine without selecting only attractive examples. The individual figures retain more detail than these compact pages.

![Machines 1-1 through 1-7](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/contact_1.png)

![Machines 1-8 through 2-6](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/contact_2.png)

![Machines 2-7 through 3-2, including 3-10 and 3-11](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/contact_3.png)

![Machines 3-3 through 3-9](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/contact_4.png)

## 7. What I would change in the investigation

1. **Keep this feature head in the shortlist.** The matched comparisons and real fault peaks justify it. Do not discard it on the strength of one macro calibrated F1.
2. **Work first on the clear calibration cases:** machine-2-8, machine-3-11, machine-3-4, and machine-1-6. Keep weights fixed and test threshold transfer on disjoint nominal blocks. The existing nominal tail and extreme quantile are a thin basis for a stable threshold; simply retuning on test labels would not establish a usable calibration method.
3. **Define whether the intended alarm is event onset or full fault duration.** If onset is the operational goal, evaluate event coverage, grouped false alarms and detection delay honestly. Do not turn peak hits into point-adjusted F1 and call localization solved. Final replay scores can use later overlapping windows, so this audit does not establish causal online latency.
4. **Treat duration errors separately:** inspect the raw dense score map, target/context influence and post-event nominal telemetry. Test whether a different score reduction preserves short peaks without extending false alarms, holding weights and window labels fixed.
5. **Use weak-separation controls:** machine-3-2, machine-3-3, machine-3-7, and machine-3-8. If the best threshold is still insufficient, improve the learned score or head there rather than continuing to search quantiles.

The evidence supports your view that many values are useful. It also explains why the evaluation is not simply wrong: **the visible peak, the usable threshold, event coverage, and alarm duration are four distinct properties**. This architecture demonstrates the first strongly on many machines; the other properties vary substantially.

## 8. Data checks and reproducibility

All 28 JSON traces cover their full saved evaluation, without truncation. Their coverage counts, scored support and truth intervals match the NPZ arrays exactly. The original HTML/JSON payload equality was verified for three samples. Window calibrated and oracle F1 were reproduced from the official score vectors.

Replay scores are close to, but not bitwise identical to, batched NPZ scores. Across all 28 machines there are **8 changed timestep decisions in total**, at the recorded calibrated thresholds; the largest calibrated timestep F1 difference is **0.000071**. One machine has one recovered-window decision difference. The trace is produced one input at a time, while NPZ scoring uses batches; that is a plausible numerical explanation, not a proven cause. These residuals do not explain the major performance gaps. Both are retained in [machine_summary.csv](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/machine_summary.csv).

Sources and details: [trace source audit](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/trace_semantics.md), [per-event evidence](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/events.csv), [peak alarm durations](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/peak_localization.csv), [matched score comparisons](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/matched_heads.csv), [HTML/JSON sample verification](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/trace_sample_verification.json). The compact `data/` archives contain the extracted final graph arrays; no model artifacts were edited.

```bash
./venv/bin/python reports/smd_cross_attention_visual_audit/analyze_traces.py
./venv/bin/python reports/smd_cross_attention_visual_audit/compare_matched_heads.py
./venv/bin/python reports/smd_cross_attention_visual_audit/verify_trace_samples.py
./venv/bin/python reports/smd_cross_attention_visual_audit/write_report.py
```
