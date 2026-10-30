# What the observed-features decision replay actually displays

Audit date: 30 October 2026. Scope: source and three saved traces for SMD `frequency_aux/cross_attention/observed_features`. This supplement does not run a model, change scoring, or overwrite existing experiment artifacts. The parent analysis covers the complete set of 28 machines and the visual review.

## The graph and the previous window ranking are different measurements

The HTML's **“Current mean by timestep”** curve is the overlap-averaged **timestep score**, not the scalar used for the earlier window ranking. For each input `[s,s+1024)`, this head emits a score map for its target half `[s+512,s+1024)`. The online accumulator adds each output value to the appropriate original timestep, increments its cover count, and divides that timestep's accumulated sum by its own count. See [placement](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/scoring.py:246) and [running means](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/scoring.py:259).

The saved suite explicitly passes the **timeseries calibrated threshold** to the replay, together with original timestep labels. [Trace call](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/experiment_suite.py:438). By contrast the previous report ranked one **mean over each 512-position target output**, labeled positive when **any** anomaly occurs in that half-window, with an independently calibrated **window threshold**. [Window score](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/scoring.py:101), [window labels and calibration](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/reconstruction_baselines.py:138).

For machine-1-1 the replay threshold is **0.24222163238873076**, whereas the window threshold is **0.07540352795273066**. These are thresholds for different score distributions. One is not a display correction for the other. A local spike can be well aligned in the timestep graph yet diluted when its entire target-half output is averaged into a window scalar. Conversely a broad error elevated by a neighboring anomaly can make normal timesteps/windows alert. The user's observation of visually useful local peaks can therefore be valid without contradicting the earlier window ranking. The graph is evidence about localization/ranking that deserves a separate, explicitly named analysis.

## Replay state and update order

The JSON stores, for every arriving input window:

- `window_start/window_end`: input/context interval;
- `checking_start/checking_end` and `indices`: output support on the original timeline;
- `scores`: **updated running means**, not new independent per-window map values;
- `counts`: how many output windows have contributed at each index so far;
- `decisions`: updated threshold decisions;
- `decision_changes`: first decisions and later flips, not only flips.

[History fields](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/scoring.py:298), [decision-change conditions](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/scoring.py:285). Updates must arrive in strictly increasing input-start order. [Order check](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/scoring.py:232). Final off-grid windows can be appended to cover the end; their final spacing can be less than the nominal stride. [Start sequence](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/decision_trace.py:36).

The renderer initializes the arrival slider at the **last recorded update**. To reconstruct the current curve it iterates updates from zero through the selected arrival and replaces each timestep's state with its most recently updated mean. It does not average the already-averaged history again. [Initial position](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/decision_trace_fragment.html:103), [state reconstruction](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/decision_trace_fragment.html:259). The lower evolution panel follows one timestep only through the windows that contribute to it; it is **not a training-epoch curve**. Smooth convergence of overlap means cannot establish stability across training epochs or random seeds. [Evolution construction](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/decision_trace_fragment.html:266).

Averages should not be reconstructed by summing every `update.scores` array. If recovering the raw contribution is necessary, use `raw = n*new_running_mean - (n-1)*previous_running_mean` at the same timestep. The verification script below uses this identity and then averages each recovered raw output map to compare against saved window scalars.

## Temporal support and the meaning of “real-time”

In the three samples, inputs are 1,024 long, output halves are 512 long, stride is 10, latency offset is zero, and the trace contains the complete evaluation sequence. Machine-1-1 covers `[512,28479)`, leaving the first 512 timesteps unscored. Ground-truth intervals before that support remain in the overview for context. They must not silently become normal scored observations or false negatives in a covered-only metric. Covered evaluation explicitly drops unscored locations. [Coverage filter](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/scoring.py:361).

The accumulated score at an old timestep can change when later complete input windows arrive. In the interior, its latest covering target half can require observations up to roughly **511 timesteps after it**; exact delay depends on stride phase and endpoint coverage. In addition, this is an **observed-target** head: it uses the target region itself as Query input, rather than forecasting that region before observing it. [Query source](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm_cross_attention.py:235). The saved example uses full-window feature extraction, which can couple the encoded context and target regions. [Feature extraction](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm_cross_attention.py:225).

Thus a good final curve establishes retrospective localization after all contributing windows arrive. A causal operating claim additionally needs a declared decision time: first available score, a fixed update count/delay, or finalized overlap mean. Inspecting only the final replay does not establish first-arrival detection or alarm latency. The HTML correctly calls itself a replay of saved windows; the broader title “Real-time anomaly decision replay” should be read in that bounded sense. [UI explanation](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/decision_trace_fragment.html:2).

The native features are downsampled by eight; the head repeats each native score over eight input positions before aggregation. [Upsampling](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/models/lewm_cross_attention.py:303). The visually fine-grained final curve mixes shifted overlapping maps and should not be interpreted as evidence of one-timestep independent model resolution.

## Logarithmic display, labels, and normalization

The initial display uses a symmetric-log transform, `sign(x)*log1p(abs(x)/abs(threshold))`, using 1 if the threshold is zero. This is monotone: the same score/threshold ordering and classification remain unchanged. It compresses large peaks and spreads smaller values, making smaller excursions easier to inspect. It can make an enormous outlier less dominant visually, but it cannot improve F1, ranking, or oracle reachability. Axis ticks/tooltips retain original score units. [Scale default](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/decision_trace_fragment.html:18), [transform](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/decision_trace_fragment.html:180), [ticks](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/decision_trace_fragment.html:209).

Vertical extent is computed from **all recorded intermediate scores**, not solely the currently visible final curve. Changing the arrival slider or horizontal zoom does not automatically refit the y-axis to that subset. [Global extent](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/decision_trace_fragment.html:155). Comparing screenshots across machines also compares different automatic vertical scales; compare numerical signal/noise separation as well as apparent peak height.

Purple bands are original **ground-truth anomaly intervals**, with half-open `[start,end)` semantics. Red/green points are **predicted anomaly/normal**, not correctness colors. Hover text supplies both prediction and ground truth. [Truth intervals](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/decision_trace.py:48), [point colors and tooltips](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/decision_trace_fragment.html:233). At full-series scale, short labeled episodes can be only a few pixels wide, while neighboring score shoulders are conspicuous; zoom before judging incident coverage or alignment.

No per-machine min–max normalization is introduced by the renderer. SMD inputs are standardized with training-file statistics before model scoring. [Input standardization](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/data/jepa_dataset.py:120). The sampled config uses `score_mode: l1`; the score map averages absolute residuals over feature channels. [Residual reduction](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/reconstruction_scores.py:16). The suite trace call supplies no score transform. [Call](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/experiment_suite.py:438). The log operation is a display transform, not a change in the model's saved numerical scores.

## Exact artifact checks and the small numerical mismatch

The verification loaded existing HTML/JSON and NPZ files, without model inference. For all three machines:

- the HTML's decoded embedded trace equals the companion JSON exactly;
- input starts, output starts/ends, cover counts, count increments, and target intervals match the NPZ artifacts;
- trace truth intervals reproduce NPZ timestep labels, and any-positive output-window labels agree;
- every saved replay decision follows its declared strict `>` threshold;
- the trace threshold equals the report's timestep threshold;
- the recorded window count equals the full saved scoring-window count; none of the three traces is truncated.

However, final replay means are **not bit-for-bit identical** to the separate saved `timeseries_scores` arrays:

| Machine | Windows | Maximum absolute score difference | RMSE | Final timestep prediction differences | Saved timestep F1 | Replay timestep F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-1 | 2747 | 0.0980647160 | 0.0024985570 | 2 | 0.4309465021 | 0.4310174514 |
| 2-3 | 2268 | 0.0082224304 | 0.0002397980 | 0 | 0.1069071711 | 0.1069071711 |
| 3-3 | 2269 | 0.0003998887 | 0.0000258201 | 0 | 0.0365659777 | 0.0365659777 |

The largest machine-1-1 absolute difference occurs at timestep 17493: NPZ score **1280.84963285006**, replay **1280.9476975661057**, a relative difference of **0.00766%**. Its median relative difference is **0.0147%**, 99th percentile **0.0748%**, maximum **0.0997%**. The two changed decisions are normal timesteps 27914–27915 close to threshold: NPZ **0.2422964357**, replay **0.2421523260**, threshold **0.2422216324**. This changes F1 by only **0.00007095**. In all three samples the raw per-window means recovered from replay have **zero changed window decisions** at the saved window threshold and exactly reproduce calibrated window F1.

The source runs model scoring again to generate the trace, **one window per update**, whereas the primary score archive uses batched scoring (sampled config `score_batch_size: 256`). [Single-window trace scoring](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/scoring.py:236), [batched primary scoring](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/utils/experiment_suite.py:367). Different batch execution/numerics is a plausible explanation for these small residuals, but the audit does not prove that cause. Historical model/source/artifact differences cannot be excluded without a controlled rescore, which was not performed. The appropriate conclusion is close numerical agreement with a small decision effect, **not exact equivalence** and not a labeling/placement mismatch in these samples.

Reproduce with `./venv/bin/python reports/smd_cross_attention_visual_audit/verify_trace_samples.py`. [Detailed numeric evidence](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/reports/smd_cross_attention_visual_audit/trace_sample_verification.json) includes score-error quantiles, changed points, geometry checks, and recovered window-score differences. Sample directories: [machine-1-1](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features/machine-1-1.txt/jepa_cross_attention_observed_features/decision_trace.html), [machine-2-3](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features/machine-2-3.txt/jepa_cross_attention_observed_features/decision_trace.html), [machine-3-3](/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/results/smd/lewm_encoder/frequency_aux/cross_attention/observed_features/machine-3-3.txt/jepa_cross_attention_observed_features/decision_trace.html).

## What a convincing graph does and does not establish

Aligned score elevations are useful evidence that the representation reacts to at least some real faults; they should not be dismissed because the clean threshold is poor. Check whether **most incidents**, including faint/short ones, rise above neighboring normal regimes, and whether broad normal elevations would create an unacceptable alarm burden. A single striking peak can dominate a whole-series view while several other incidents remain indistinguishable from normal scores.

Oracle thresholding asks whether **one constant cutoff** separates the complete sequence. A reader can mentally adapt a cutoff by time, recognize a local baseline, merge an event, ignore shoulders, and tolerate delayed responses. That can make the plot visually compelling even when a constant-threshold F1 is limited. Those visual operations might motivate a valid detector extension—local calibration, episode grouping, or persistence—but they are additional policies requiring clean-data design and prospective evaluation.

The final curve alone cannot establish ease of automatic calibration, epoch/seed stability, causal early detection, a false-alarm budget, or generalization. It can support localization and motivate a matched numerical audit. Keep the window-only results as the answer to the original window task, and explicitly name the additional timestep/event evidence used to investigate the user's graphical observation.
