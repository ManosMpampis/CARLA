---
title: TF-Scout dual-stream losses and action-conditioned variant
labels: [ready-for-agent]
created: 2026-09-03
branch: jepa
spec-source: DESIGN_TSAD_DUALSTREAM.md, DESIGN_TF_SCOUT_LOSSES.md
---

# SPEC: TF-Scout dual-stream losses and action-conditioned variant

## Problem Statement

As a time-series anomaly researcher I have a dual-stream detector design (time pathway scouts, frequency pathway searches, steered and compared part-vs-whole) but no committed loss-per-branch plan, so I cannot run the head tournament or test whether a YOLO-style proposal signal can condition reconstruction without breaking honest evaluation.

## Solution

Finalize TF-Scout v2 as a fixed trunk with interchangeable training-only focus heads and four scored pluggable heads, per-branch committed losses, open-loop baseline plus closed-loop action-conditioned variant, per-machine normal-only training, dense canonical output, selected purely by tournament on honest metrics.

## User Stories

1. As a researcher, I want the time pathway to act as a focus module with training-only heads pruned at inference, so that localization improves without paying inference cost.
2. As a researcher, I want separate reconstruction-focus and box-focus auxiliary heads plus one combined criterion, so that I can ablate each focus signal alone.
3. As an experimenter, I want all four head families (detection, reconstruction, energy, metric-learning) as independent Head + Loss pairs, so that I can run the full tournament.
4. As an experimenter, I want H1 + H4 treated as primaries with H2/H3 as fused channels, so that effort concentrates where I stated interest without losing baselines.
5. As a researcher, I want per-token view-KL as the default distribution-alignment channel with a window-VAE fallback, so that sub-window disagreement is scored with a collapse-safe fallback.
6. As a researcher, I want per-level tiny VAEs gated behind annealing + free-bits, so that I can test the sub-window VAE idea without destabilizing the trunk.
7. As an experimenter, I want per-machine single-center metric learning as default with a multi-component variant, so that multi-regime machines are modeled without cross-machine leakage.
8. As a researcher, I want forecasting dropped and horizons reinterpreted as masked-part indices, so that training matches the reconstruction-only preference.
9. As a researcher, I want part-reconstruction scored with MSE plus a non-negative shape divergence on proposed parts only, so that spikes and morphology shifts are both caught cheaply.
10. As a researcher, I want interval detections trained only on synthetic boxes with hard mining, so that proposals exist without touching test labels.
11. As a world-model user, I want a disposable projector before the isotropy regularizer, discarded for scoring, so that the latent geometry constraint does not pollute anomaly scores.
12. As a world-model user, I want one anti-collapse mechanism per run (isotropy regularizer, EMA teacher, or codebook), so that failure sources stay isolated.
13. As a researcher, I want the open-loop baseline (propose, mask, reconstruct, no feedback) built first, so that the closed-loop variant has something to beat.
14. As a researcher, I want the closed-loop variant (detached proposals into action embedding into conditional predictor/decoder) as a separate arm, so that the YOLO-as-action proposition is tested fairly.
15. As an experimenter, I want detached proposals with gradient-conflict monitoring, so that feedback cannot collapse into explaining anomalies away.
16. As an evaluator, I want the per-timestep score map as the canonical output with intervals secondary, so that the frozen metrics contract never changes.
17. As an evaluator, I want normal-only training with train-only quantile thresholds and a mandatory no-training baseline, so that honest numbers stay honest and point-adjust stays a separate comparability column.
18. As an experimenter, I want per-machine training always for headlines, so that results follow the official protocol.
19. As a reviewer, I want the corrected divergence citations (shape divergence vs temporal distortion as distinct terms), so that the loss choices are defensible.

## Implementation Decisions

- Trunk fixed for tournament: STFT TF-grid representation, dual pyramids with top-down fusion, every-level feature-wise modulation plus single bottleneck cross-attention, one-directional time-to-frequency steering.
- Part-vs-whole comparison fixed to three axes (part-to-global, fine-to-coarse, part-to-part) with two operators (learnable cross-scale prediction, parameter-free vector comparison); every mismatch map is a named score channel for fusion.
- Time pathway carries two training-only auxiliary heads (reconstruction focus, box focus); combined criterion is pure summation with per-member enable flags.
- Predictor predicts masked parts, not futures; decoder reconstructs proposed parts; full-window forecasting explicitly rejected.
- Shape loss lives only inside proposed parts (pointwise term plus shape-divergence term; temporal-distortion term as ablation); complexity contained by part masking.
- Box geometry learned only from synthetic injections; dense map stays canonical so box bias cannot inflate honest scores.
- Metric head uses per-machine normalized center plus variance guard plus inverse-distance push on synthetics; multi-component mixture energy kept as variant; balanced-cluster objectives rejected.
- Energy head and time-frequency consistency channel kept as tournament-only variants.
- Closed-loop action path: stop-gradient proposals into small box-to-action embedding into zero-initialized conditional modulation of predictor/decoder; separate schedule from open-loop; promotion requires beating open-loop plus no-training baseline.
- Budgets: order 10⁵–10⁶ params, single-GPU mixed-precision training, full-precision scoring, eval-mode validation statistics, CPU-plausible inference.

## Testing Decisions

- Good tests check external behavior at the highest seam (registry-built trunk plus criterion trains, dense outputs aggregate cover-count-aware, metrics contract holds), not internal tensor shapes.
- Modules under test: trunk construction, each Head + Loss pair alone, each combined criterion, open-loop vs closed-loop wiring, per-machine calibration thresholds.
- Prior art: synthetic tracer plus resume script, scorer handoff (overlap aggregation plus metric contract), arms script, stages script (masking/corpus/adapt/variants), calibration script (train-only thresholds), single-machine smoke chain end-to-end.

## Out of Scope

- Transformer-first trunk; joint-corpus headlines; full-window forecasting; test-label-touched thresholds; point-adjust as headline; memory-bank methods (deliberately demoted variant); learned filterbank front-end (deferred); bidirectional steering as default; metrics-stack changes; datasets/results symlinks touched or committed.

## Further Notes

- Source designs: dual-stream v1 left loss open deliberately; v2 loss plan closes it after grill rounds 1–2 with all-arm tournament retained.
- Known evaluation hazard carried forward: prediction-error-only scoring has hit chance level under fair protocols in the literature — hence at least two fused channels plus no-training baseline are mandatory, not optional.
