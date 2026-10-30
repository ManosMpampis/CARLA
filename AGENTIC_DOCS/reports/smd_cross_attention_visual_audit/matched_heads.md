# Matched saved-score comparison: observed features, Query, input, and no-FiLM features

The `frequency_aux/cross_attention/observed_features` model contains useful anomaly information. The saved score archives do **not** support dismissing it as substantially worse than the observed-Query model. Its window results are close to Query and no-FiLM features and generally stronger than observed input. At the same time, visible peaks are not sufficient evidence that its normal-only threshold transfers well or that every anomaly event is detected reliably.

## Scope and reproduction

This audit uses all 28 SMD machines for each of four base, validation-selected families, 112 score archives total. Every comparison has input length 1024, scored target span `[512,1024)`, stride 10 and identical saved window labels, scored start/end indices, timeline labels and coverage masks. The script asserts those equalities and verifies that saved calibrated predictions and F1 agree with their saved thresholds and metrics. All assertions pass.

The primary scope remains the requested **overlapping window task**. To test the visual impression of the full-timeline HTML, an additional explicitly descriptive analysis uses continuous timeline scores. It does not replace the window ranking or mix the two sets of metrics. Timeline ordinary AP/ROC below are computed on continuous scores at covered timesteps only. No non-overlap results enter this audit.

The timeline source here is the saved `scores.npz`, generally produced with scoring batch size 256. The parent's separate visual audit uses the decision-trace JSON, produced at batch size 1; small numerical differences can occur. These are not fresh inference results. This report makes no claim that the two exports are bit-identical.

Reproduce with:

```bash
./venv/bin/python reports/smd_cross_attention_visual_audit/compare_matched_heads.py
```

Sources: [comparison script](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/compare_matched_heads.py), [complete per-machine data and pairwise comparisons](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/matched_heads.json), [CSV](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/matched_heads.csv). Every machine row records the exact original `metrics.json` and `scores.npz` paths.

Family abbreviations:

- **Features:** `lewm_encoder/frequency_aux/cross_attention/observed_features`.
- **Query:** `lewm_encoder/frequency_aux/cross_attention/observed_query`.
- **Input:** `lewm_encoder/frequency_aux/cross_attention/observed_input`.
- **No-FiLM features:** `lewm_encoder/frequency_aux_no_film/cross_attention/observed_features`.

## Window comparison across the same 28 machines

All numbers are equal-machine means, not a pooled mixture dominated by long machines.

| Family | Calibrated F1 | Oracle F1 | VUS-PR | VUS-ROC |
|---|---:|---:|---:|---:|
| Features | 0.4046 | 0.6489 | 0.5655 | 0.7024 |
| Query | 0.4150 | 0.6404 | 0.5692 | 0.7059 |
| Input | 0.3450 | 0.5852 | 0.4980 | 0.6512 |
| No-FiLM features | 0.4060 | 0.6564 | 0.5647 | 0.6984 |

Features beats Query on oracle F1 on **17 machines**, loses on 10 and ties on 1. Its calibrated F1 wins/losses are **14/14**. Its macro VUS differences from Query are only −0.0037 PR and −0.0035 ROC. These are small descriptive differences, not evidence of a reliable advantage for either family across training seeds.

Features beats Input on calibrated F1 on **20/28**, oracle F1 on **24/28**, VUS-PR on **21/28**, and VUS-ROC on **19/28**. Features and no-FiLM features are also close at the macro level, with substantial machine-specific reversals; that does not support a universal claim that FiLM improves or harms this head.

Thus the correct broad conclusion is: **Features is a competitive observed cross-attention head in this matched comparison; its main concern is reliable decisions from its score, and all three leading heads share that concern.** It is not the strongest family in the whole repository merely because it is competitive among these four heads.

## Does the full timeline contain useful peaks?

Yes. Continuous timeline discrimination is often substantially stronger than a low calibrated F1 suggests.

| Family | Ordinary timeline AP | Ordinary timeline ROC AUC | Calibrated timeline F1 | Nominal timestep FPR | Event-hit rate |
|---|---:|---:|---:|---:|---:|
| Features | 0.4632 | 0.8811 | 0.2308 | 0.2122 | 0.7033 |
| Query | 0.4532 | 0.8666 | 0.2238 | 0.2164 | 0.7137 |
| Input | 0.4317 | 0.8210 | 0.2125 | 0.1597 | 0.9505 |
| No-FiLM features | 0.4570 | 0.8734 | 0.2402 | 0.2023 | 0.6658 |

Here **AP is ordinary average precision on raw covered timeline scores**, not VUS-PR and not window AP. ROC likewise uses raw covered timeline scores. Event hit means at least one threshold crossing within an annotated event, with no point adjustment. FPR is the fraction of covered nominal timesteps above threshold, not false alert episodes per hour. It can count many consecutive crossings from one persistent false alarm. Event-hit rates are averaged per machine, not pooled across events.

Features has slightly higher macro timeline AP and ROC than Query; it wins AP on 18/28 and ROC on 15/28. This supports the visual impression that useful signal remains. But an average calibrated nominal-timestep FPR around **21%** is a substantial decision burden, and an event-hit rate around 70% does not mean it catches every event. A clear peak can be useful to an analyst while the fixed threshold still produces too many background alerts or misses other events.

Input's event hit is around 95%, despite weaker macro AP, ROC, and window F1. This is not a contradiction: catching one point of an event says little about alert precision, coverage of the event, or window-level discrimination. Long events and broad weak threshold crossings can make any-hit recall high. Both event hit and false-alert burden must be inspected.

At each machine's **test-oracle timeline-F1 threshold**, Features averages timeline F1 **0.5022**, nominal FPR **0.0518**, and event hit **0.5674**. The calibrated-to-oracle F1 gap is large, supporting threshold-transfer concerns. The lower oracle event hit also shows that maximizing timestep F1 may sacrifice short events to improve precision on longer ones. This is an oracle threshold selected with test labels, not a deployable normal-only choice.

## Machines that explain the disagreement

### machine-3-11: excellent window ranking, poor transferred calibration

Features window VUS-PR/ROC are **0.9335/0.9832**, oracle window F1 is **0.9174**, but calibrated window F1 is **0.0947**. Its timeline AP/ROC are **0.3678/0.9584**, while the calibrated threshold flags **61.8%** of nominal covered timesteps. The oracle timeline F1 rises from **0.0224** calibrated to **0.4961**.

This is a concrete case where seeing a useful score shape and seeing a poor calibrated metric can both be correct. Window discrimination is excellent; the fixed nominal-tail threshold does not transfer. Input happens to have higher calibrated window F1 (**0.2039**) but much worse oracle window F1 (**0.6984**) and VUS-PR (**0.5414**), illustrating why one operating point should not determine the ranking of the score itself.

Data: [Features metrics](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features/machine-3-11.txt/jepa_cross_attention_observed_features/metrics.json), [Features scores](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features/machine-3-11.txt/jepa_cross_attention_observed_features/scores.npz).

### machine-3-4: Features genuinely outperforms Query, but calibration nearly always alarms

Features timeline AP is **0.8146**, compared with Query **0.4577**; timeline ROC is **0.9134** versus **0.5436**. Features window oracle F1 is **0.4922**, versus Query **0.3657**, and VUS-PR/ROC are **0.4308/0.6996**, versus **0.2467/0.4525**. This is a substantial advantage for the feature target on this saved machine.

Yet both calibrated timeline thresholds flag about **90% of nominal timesteps**. Both achieve event hit 1.0, which is therefore weak reassurance. Features' timeline oracle F1 is **0.7979**, but the threshold maximizing that score hits only **25% of annotated events**. Different event lengths make pointwise F1 and event count very different objectives. Input is better still in ordinary timeline AP on this machine (**0.8971**), so even a genuine Features-over-Query success does not make Features best on every criterion.

Data: [Features metrics](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features/machine-3-4.txt/jepa_cross_attention_observed_features/metrics.json), [Features scores](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features/machine-3-4.txt/jepa_cross_attention_observed_features/scores.npz).

### machine-3-7: Query contains a stronger useful signal

Features and Query both hit all annotated events at the calibrated threshold, but nominal timestep FPR is about **31.6%/30.8%**. Query has better ordinary timeline AP (**0.6446** vs **0.3996**) and oracle timeline F1 (**0.6914** vs **0.4847**). Their timeline ROC difference is smaller (**0.9207** vs **0.8897**), consistent with two scores that both rank many nominal points below anomalies but differ substantially in the high-score precision region. Window oracle F1 is also better for Query (**0.3562** vs **0.3248**). Features' visible peaks therefore coexist with a real comparative weakness here.

Data: [Features metrics](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features/machine-3-7.txt/jepa_cross_attention_observed_features/metrics.json), [Query metrics](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_query/machine-3-7.txt/jepa_cross_attention_observed_query/metrics.json).

### machine-2-5: Query wins window ranking; feature steering still helps relative to no-FiLM

Query window VUS-PR/ROC are **0.8637/0.8786**, versus Features **0.6666/0.7201**. Query's calibrated/oracle window F1 are **0.4225/0.7690**, versus **0.3214/0.7315**. These are meaningful saved-score differences. But Features is stronger than no-FiLM features on this machine: no-FiLM's calibrated/oracle window F1 are **0.1596/0.6482** and VUS-PR/ROC **0.5129/0.5556**. FiLM lineage cannot be described as uniformly bad from a few poor examples.

Data: [Features metrics](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features/machine-2-5.txt/jepa_cross_attention_observed_features/metrics.json), [Query metrics](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_query/machine-2-5.txt/jepa_cross_attention_observed_query/metrics.json).

### machine-1-2: no-FiLM features is clearly better in this comparison

No-FiLM features has calibrated/oracle window F1 **0.7516/0.7677**, against Features **0.5791/0.5845**; VUS-PR is **0.7972** against **0.5978**. Ordinary timeline AP is **0.5907** against **0.4626**. Features is comparatively weaker here even though its timeline ROC is high (**0.9642**). This is a useful paired case for examining how the transferred representation differs rather than treating all feature-target runs as one success/failure pattern.

Data: [Features metrics](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features/machine-1-2.txt/jepa_cross_attention_observed_features/metrics.json), [No-FiLM metrics](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux_no_film/cross_attention/observed_features/machine-1-2.txt/jepa_cross_attention_observed_features/metrics.json).

## Interpretation and limits

1. **Keep observed Features on the shortlist for investigation.** The matched comparison does not support a broad claim that Query is decisively better. Features has useful continuous-score discrimination and stronger results than Input on most of these machines.
2. **Separate score quality from calibration.** machine-3-11 and machine-3-4 deserve calibration and nominal-regime inspection before redesigning the representation. Their large calibrated/oracle gaps are empirical; the reason the nominal distribution shifts still requires inspecting data and normalization.
3. **Inspect the actual failing machine.** machine-3-7 and machine-2-5 show genuine comparative weaknesses, while machine-3-4 shows a clear advantage over Query. A single averaged rank hides these reversals.
4. **Make event usefulness explicit.** A visually attractive peak, event hit, timestep F1 and mean-window F1 answer different questions. Choose whether deployment values any timely warning, sustained localization, few false alert episodes, or a window verdict; report the relevant burden with it.
5. **Do not infer causation from this paired archive audit.** Geometry and labels are controlled, but targets, learned representations and selected epochs differ. There is one saved training realization per machine/family here, with no training-seed uncertainty estimate. Near-ties in macro scores are not statistical equivalence claims.

The source archives, thresholds and per-machine results in the accompanying JSON allow every numerical claim above to be checked. No model was retrained, re-scored or modified.
