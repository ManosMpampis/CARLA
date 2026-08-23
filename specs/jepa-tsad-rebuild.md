---
title: JEPA-based TSAD rebuild (convolutional world-model pipeline)
labels: [ready-for-agent]
created: 2026-08-23
branch: jepa
spec-source: DESIGN_TSAD_JEPA.md, RESEARCH_LEWORLDMODEL.md, RESEARCH_JEPA_WORLD_MODELS.md, RESEARCH_TS_AD_HEADS.md
---

# SPEC: JEPA-based TSAD rebuild

## Problem Statement

The repo's current two-stage detector (contrastive pretext → window classification) learns *a* representation of normal windows and classifies whole windows, but it cannot say **where inside a window** something became strange, and its training signal depends entirely on contrastive pairs. The owner wants to rebuild the project around a JEPA / LeWorldModel framework: a self-supervised model that learns *how to extract representations* of time-series windows and flags anomalies as **latent prediction surprise localized at sub-window granularity** — while remaining small and fast enough to score windows continuously on machines without CUDA. The old implementation must stay available on other branches, published SOTA numbers on SMD/PSM are the bar to beat, and the evaluation must be honest (train-only thresholds, point-adjust reported separately rather than silently inflating results).

## Solution

Replace the contrastive pipeline with a convolutional-pyramid JEPA trained by dense latent prediction: an encoder pyramid produces latents at four time scales (sub-windows of 1/2/4/8 steps); causal predictors learn to predict each token's near-future latents from strictly past tokens; anomalies are positions where prediction diverges from what the target branch actually encodes. Anti-collapse is pluggable (SIGReg primary; EMA teacher and soft-codebook arms). Synthetic injected anomalies (`SubAnomaly`) stop being contrastive negatives and become an unlabeled calibrator: thresholds come from clean-train plus injected-probe score distributions, never from test labels. All architectures under test are selected purely by YAML configs; evaluation reuses the existing metrics stack verbatim (regular F1, point-adjust F1, affiliation, event-F1, MCC, R_AUC/VUS) fed by a new per-timestep score emitter. Training supports mixed precision; inference is CPU-class (measured 3 ms/window against a 5-second budget).

## User Stories

1. As a TSAD researcher, I want to pretrain a JEPA encoder on a single machine's training windows so that I follow the official SMD protocol where each machine is its own training set.
2. As a TSAD researcher, I want to optionally pretrain jointly across all 28 SMD machines so that I can test whether cross-machine generality helps before adapting per machine.
3. As a TSAD researcher, I want a second training stage that adapts pretrained weights to the target machine so that general time-series knowledge transfers to the machine being monitored.
4. As a TSAD researcher, I want to choose between frozen-encoder and full-finetune adaptation so that I can measure whether frozen features are the bottleneck.
5. As a TSAD researcher, I want masked-block latent prediction implemented as a first-stage objective so that the model first learns how time series behave before learning my task.
6. As a TSAD researcher, I want mask-free forward prediction (horizons k=1,2) as the primary task objective so that temporal continuity is never disrupted during task training.
7. As a TSAD researcher, I want SIGReg regularization as the primary anti-collapse mechanism so that I avoid EMA-teacher instability reported for continuous self-distillation on time series.
8. As a TSAD researcher, I want an EMA-target-encoder arm so that I can isolate which collapse-prevention mechanism works best on my data.
9. As a TSAD researcher, I want a soft-codebook arm so that I get regime-prototype signals (distance-to-prototype, attention entropy) as additional anomaly evidence.
10. As a TSAD researcher, I want TCN and GRU predictor variants selectable by config so that I can compare parallel versus recurrent predictive heads.
11. As a TSAD researcher, I want a transformer-encoder comparison arm kept alongside the conv pyramid so that I can quantify whether attention capacity earns its cost.
12. As a TSAD researcher, I want anomaly scores computed per sub-window at each pyramid level and fused by mean so that one score reflects both fine-grained breaks and coarse trend breaks.
13. As a TSAD researcher, I want fusion weights later calibrated on injected-anomaly probes so that level weighting becomes data-driven without labels.
14. As a TSAD researcher, I want injected synthetic anomalies used only as held-out probes/calibrators so that the representation is never shaped by synthetic quirks.
15. As a TSAD researcher, I want thresholds derived exclusively from training statistics so that no information from labeled test data leaks into detection decisions.
16. As a benchmark reviewer, I want window-level AUROC/AP reported as the headline honest metric so that my numbers are comparable without point-adjust inflation.
17. As a benchmark reviewer, I want point-level F1 reported without point-adjust as the secondary metric so that localization quality is visible honestly.
18. As a benchmark reviewer, I want point-adjust metrics in a clearly separated column so that I can still position results against PA-based published numbers.
19. As a benchmark reviewer, I want the existing metrics dictionary (affiliation, event-F1, MCC, R_AUC, VUS-ROC/VUS-PR) returned unchanged in format so that aggregation scripts keep working.
20. As a benchmark reviewer, I want a reference table of published SMD/PSM numbers with provenance flags so that I know exactly which bar I am beating and how it was measured.
21. As a monitoring operator, I want single-window CPU inference to complete well within five seconds so that the detector can run continuously on non-CUDA machines.
22. As a monitoring operator, I want the trained model loadable into a fresh process for scoring so that deployment does not require the training code path.
23. As a monitoring operator, I want overlapping sliding-window scores aggregated consistently per timestep so that points covered by many windows are not double-counted or dropped.
24. As a repo maintainer, I want every architecture arm defined by a config file so that adding a variant means writing YAML, not copying scripts.
25. As a repo maintainer, I want component construction behind registry factories keyed by config names so that arms share one wiring convention.
26. As a repo maintainer, I want checkpoints and resume behavior in the existing format so that interrupted runs continue and old tooling stays valid.
27. As a repo maintainer, I want mixed-precision training behind a config flag so that GPU memory and speed improve without touching scoring precision.
28. As a repo maintainer, I want all new logic expressed as classes with methods so that components remain testable, composable, and readable.
29. As a repo maintainer, I want the legacy contrastive training loop, unused losses, and superseded configs deleted after cutover so that the branch stays navigable.
30. As a repo maintainer, I want the existing contrastive dataset classes kept intact but unused so that prior experiments remain reproducible from this branch.
31. As a repo maintainer, I want TensorBoard logging of losses, scores, and the model graph so that training health is inspectable without reruns.
32. As a repo maintainer, I want previously produced pretext summary CSVs preserved in a dedicated directory so that historical evidence is not lost by the rebuild.
33. As an agent implementing this spec, I want a tiny synthetic-data config that exercises every stage end-to-end quickly and deterministically so that I can verify the whole pipeline without datasets or GPUs.
34. As an agent implementing this spec, I want a real-data smoke config on one machine so that acceptance is demonstrable on actual SMD telemetry.
35. As an agent implementing this spec, I want validation loss (latent-prediction) as the checkpoint-selection signal so that selection optimizes what the model actually does.
36. As a TSAD researcher, I want validation-time statistics computed in eval mode so that normalization-state leakage cannot inflate my selection metric.

## Implementation Decisions

- **Branching**: dedicated rebuild branch (`jepa`); previous implementation remains intact on existing branches. Historical pretext summaries relocated into a dedicated local directory; `results/` output tree never deleted.
- **Harness shape**: one entrypoint dispatches stages (`pretrain`, `adapt`, `score`) driven entirely by merged env+experiment configs, mirroring the repo's existing create_config/factory pattern. Arms = YAML files; no copied driver scripts.
- **Model**: convolutional pyramid encoder with four levels; causal predictors (TCN default, GRU optional) forecast k∈{1,2} horizons per level. Prototype-measured geometry (from the working preview, not aspirational):

  ```
  x:(B,38,256) → L0(32ch,T=256,sub-window=1 step)
               → L1(32ch,T=128,2 steps) → L2(64ch,T=64,4) → L3(96ch,T=32,8)
  predictor_ℓ: causal dilated conv (k=3, dilations 1/2/4) → (B,2,Dℓ,Tℓ)
  params ≈ 0.30M; CPU ≈ 3 ms/window
  ```

- **Loss**: dense L1 between predicted and stop-gradient target latents over ALL tokens × levels × horizons (context tokens included, V-JEPA 2.1 rationale), plus λ·SIGReg per token (λ≈0.1, sliced Epps–Pulley statistic). Targets layer-normed in the EMA arm.
- **Anti-collapse registry**: `sigreg` | `ema` | `codebook`. SIGReg projections never enter the anomaly-score path. Codebook = K prototypes with soft-attention routing and k-means warmup init.
- **Two-stage corpus semantics**: stage-A `corpus: joint|single`; official/headline protocol = per-machine (SMD treats machines as separate training sets); joint is exploratory. PSM is always its own corpus.
- **Scoring**: per-position ‖predicted − target‖₁ per level → mean-fused → per-timestep arrays with start/end indices → overlap-aware aggregation → thresholds from clean-train + injected-probe quantiles. Calibration uses `SubAnomaly` injections as the sole probe source.
- **Evaluation seam**: the new Scorer emits `(scores, start_idxs, end_idxs)` consumed by the EXISTING metrics functions unchanged; metric dictionary keys and semantics are frozen as-is.
- **Checkpointing**: existing resume/checkpoint format extended minimally (predictor/target state, stage metadata); validation latent-prediction loss replaces clustering metrics as the selection signal; clustering metrics removed from selection paths.
- **Precision**: AMP (bf16 autocast + GradScaler) training-only behind config flag; scoring always fp32; validation statistics computed in eval mode.
- **Code style**: classes with methods throughout (`JEPAModel`, encoders, predictors, `SIGReg`, `EMAWrapper`, `SoftCodebook`, `MaskingCollator`, `JEPADataset`, `Trainer`, `Calibrator`, `Scorer`). New data class coexists with — does not modify — the four existing dataset classes, which become unused but functional.
- **Deletion manifest** executed after end-to-end cutover: triplet/classification losses, old entry scripts and drivers, superseded config trees, clustering-evaluation selection code. Untouched: metrics package, Logger, GradientMonitor, checkpoint utilities, collate, SMD/PSM loaders, augmentation/SubAnomaly machinery.
- **Known-bug avoidance**: new strided blocks implement their own padding because the repo's same-padding helper assumes stride 1 (documented; legacy backbone unaffected).
- **Build order** (each phase ends verifiable through the primary seam):
  1. *Foundation*: architecture registry, legacy backbone module rename with import updates, EMA wrapper, SIGReg loss class, JEPA criterion class(es), MaskingCollator, JEPADataset.
  2. *Training*: Trainer class (AMP + existing checkpoint/resume format), entrypoint stages `pretrain`/`adapt`.
  3. *Scoring & evaluation*: Calibrator → Scorer → per-timestep arrays into the untouched metrics stack; experiment configs for arms J1/J2/J3/J4/JT.
  4. *Cutover*: execute the deletion manifest, publish the reference-numbers document, run the real-data smoke config end-to-end and verify TensorBoard + metrics output.

## Testing Decisions

- **What makes a good test here**: external behavior through the highest seam — running the stage entrypoint with a config and asserting on artifacts (checkpoint resumability, TensorBoard events, metrics dictionaries with expected keys/ranges) — never internal tensors or private helpers.
- **Primary seam (one)**: CLI stage runner driven purely by merged config. Two canonical test configs: (a) synthetic tiny data, deterministic, CPU-only, minutes-fast, exercising every stage; (b) real machine-1-1 short schedule as acceptance smoke.
- **Single auxiliary assertion point**: the scorer→evaluator handoff — per-timestep arrays plus start/end indices must flow into the untouched metrics stack and return the frozen metric-dictionary contract. This is also where overlap-aggregation correctness is checked (a point covered by N windows appears once per timestep, correctly weighted).
- **Prior art**: the repo has no unit-test framework; its verification convention is short training-config runs (documented in AGENTS.md) plus the earlier synthetic-value audits of checkpoint logic recorded in WORKFLOW_PRETEXT.md. Tests follow that convention instead of introducing a framework.
- **Regression guards**: old-format checkpoints remain loadable/resumable; metrics dictionary keys unchanged; deleted-module imports fail loudly only in legacy entrypoints (which are themselves deleted).

## Out of Scope

- Reproducing published baselines locally (reference numbers are harvested from papers/benchmarks with provenance flags; owner manually verifies afterward).
- Hybrid contrastive+JEPA margin objectives (deferred ablation).
- Deeper H-JEPA-style hierarchies beyond the fixed four levels; action-conditioned predictors (no action channel exists in SMD/PSM).
- Distributed/multi-GPU training; any new visualization beyond TensorBoard.
- Changes to dataset parsing/normalization semantics of the SMD/PSM loaders.
- DataLoader worker-seed reproducibility overhaul (known limitation, documented, not addressed here).
- Modifying the four legacy dataset classes or the legacy metrics implementations.

## Further Notes

- Novelty positioning (grounded in RESEARCH_LEWORLDMODEL.md §5): no published work combines dense all-token latent prediction + sub-window-level MTS anomaly scoring + synthetic-anomaly threshold calibration; this build stakes that combination.
- Negative controls from the literature are treated as design constraints: raw prediction error alone has hit chance-level under fair protocols elsewhere, hence ≥2 fused score types and mandatory no-training baselines in reports.
- The independent LeWorldModel reproduction found BatchNorm-inflated validation losses (up to 300×) and undocumented config dependencies; therefore validation statistics are computed in eval mode and configs are kept minimal/explicit.
- Measured feasibility anchors from the prototype: 297,600 parameters, 3.0 ms/window CPU (328 windows/sec) against the 5-second online budget; TensorBoard graph preview lives under the arch_preview version directory.
