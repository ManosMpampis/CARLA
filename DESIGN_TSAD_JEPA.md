# DESIGN: JEPA-based TSAD rebuild (v1 — agreed after grilling session, 2026-08-23)

Status: **AGREED SPEC**. Every decision below was settled interactively; deviations require revisiting the relevant item. Research grounding: `RESEARCH_LEWORLDMODEL.md`, `RESEARCH_JEPA_WORLD_MODELS.md`, `RESEARCH_TS_AD_HEADS.md`.

---

## 0. Decision ledger

| # | Decision | Choice |
|---|---|---|
| D1 | Branch strategy | Dedicated `jepa` branch; old implementation stays intact on `main`/`clean_branch`/`before_changes`. `results/` never deleted; `pretext_summary*.csv` relocated to `pretext_results/`. |
| D2 | Rebuild shape | Multi-architecture harness: registry-driven factories keyed by YAML (`get_*` pattern mirroring `utils/common_config.py`). One YAML per architecture under `configs/jepa/<arm>/`; entry script `carla_jepa.py` with `stage: {pretrain \| adapt \| score}`. |
| D3 | Collapse prevention | Arms: **SIGReg** (primary), **EMA target encoder**, **soft codebook** (SC-JEPA-style fallback). All selectable via `anti_collapse:` key. SIGReg projections stay out of the anomaly-score path. |
| D4 | Evaluation protocol | Both detection and localization. Headline = honest window-level AUROC/AP + point-level F1 **without** point-adjust; point-adjust column reported separately ONLY for literature comparability (most published numbers use it). Thresholds from TRAIN statistics, calibrated on clean-train + `SubAnomaly`-injected probe windows. Never touch test labels for tuning. |
| D5 | Evaluation tooling | Reuse `metrics/metrics.py` verbatim (regular F1 + PA-F1 + affiliation + event-F1 + MCC + R_AUC/VUS-ROC/VUS_PR) and the checking-window accumulation in `evaluate()`. Adapt input format: scorer emits per-timestep score arrays + start/end indices instead of class probabilities. `pr_evaluate_timeseries` rewritten around that seam. |
| D6 | `SubAnomaly` role | Held-out calibration/probing ONLY (not inside the loss) for v1. Hybrid margin-loss variant deferred to a later ablation. Novelty stake: first MTS-AD using injected anomalies as unsupervised threshold calibrators combined with dense latent prediction. |
| D7 | Primary objective | Mask-free multi-horizon (k∈{1,2}) dense all-token L1 latent prediction (V-JEPA 2.1-style context loss included so visible tokens carry calibrated targets). Block-masking implemented as `MaskingCollator` for stage-A and ablation arm J2 (CF-JEPA caution noted). |
| D8 | Two-stage learning | Stage A: masked-block latent prediction pretraining (learns how time series behave). Stage B: task adaptation on target machine. `stage_b.mode: frozen\|finetune`, default frozen (HEPA pattern; finetune arm measures frozen-feature bottleneck per Asleep-at-the-Wheel). |
| D9 | Stage-A corpus | `stage_a.corpus: joint\|single`. Joint = all 28 SMD machines together (exploratory arm). **Official SMD protocol treats machines as separate training sets → headline results use per-machine.** PSM gets its own corpus. |
| D10 | Model family | **NOT transformer-first.** Primary: convolutional pyramid `JEPAPyramid` (D10a). Transformer token encoder kept as comparison arm only. CPU inference constraint: comfortably fast (measured 3 ms/window vs 5 s budget). |
| D11 | Score fusion | Mean across levels first; weights later calibrated on `SubAnomaly` probes. |
| D12 | Success criteria | Beat published numbers (F1, VUS, etc.) on SMD **and** PSM incl. point-adjust-reported methods. Broad comparison table in `REFERENCE_NUMBERS.md` with provenance flags (paper-reported vs benchmark-reproduced). Owner manually verifies results after experiments. |
| D13 | Checkpoint selection | Validation latent-prediction loss. Clustering metrics (Silhouette/CH/DB) removed from selection; not kept even as diagnostics requirement. |
| D14 | Mixed precision | Config flag `amp: true` → bf16 autocast + GradScaler during training only; scoring always fp32. |
| D15 | Code style | All new components are classes with methods (no bare-function pipelines): `JEPAModel`, encoders, predictors, `MaskingCollator`, `Scorer`, `Calibrator`, `Trainer`, loss classes. |

## 1. Architecture (as built and measured)

```
input window x: (B, 38 sensors, 256 timesteps)
        │
   ┌────▼─────┐
   │ stem k=7 │─────────► L0: (B, 32, 256)   sub-window = 1 step   ◄─ finest detail
   └────┬─────┘              │
   ┌────▼─────┐              │
   │ conv ÷2  │─────────► L1: (B, 32, 128)   sub-window = 2 steps
   └────┬─────┘              │
   ┌────▼─────┐              │
   │ conv ÷2  │─────────► L2: (B, 64, 64)    sub-window = 4 steps
   └────┬─────┘              │
   ┌────▼─────┐              │
   │ conv ÷2  │─────────► L3: (B, 96, 32)    sub-window = 8 steps   ◄─ coarsest trend
   └──────────┘

each level ──► causal predictor (tcn default | gru arm), horizons k∈{1,2}
               predicts future latents from strictly-past latents

score(level ℓ, position i) = ‖ predicted − sg(actual) ‖₁   → mean-fused across levels
loss = Σ_ℓ mean‖pred − sg(target)‖₁  +  λ·SIGReg(per-token)     λ≈0.1
```

Measured (CPU, batch=1): 297,600 params; 3.0 ms/window (328 windows/sec); graph logged at `results/arch_preview/tensorboard`.

Implementation notes:
- Uses its own `StridedConvBlock` — repo's `Conv1dSamePadding` assumes stride 1 when padding and silently cancels stride>1 downsampling (latent bug, never hit by old ResNet).
- Causality: predictors are left-padded causal convs; encoder may be non-causal within the window.
- SIGReg: sliced random-projection Epps–Pulley normality statistic per token, quadrature nodes [0.2, 4], λ the single effective hyperparameter (LeWM App. A).
- EMA arm: linear-interp momentum schedule, constant 0.99925 default (V-JEPA 2 shipped config), targets layer-normed before masking/loss.
- Codebook arm: K prototypes, soft attention routing, k-means warmup init; adds distance-to-prototype + attention-entropy signals.

## 2. Experiment matrix

| Arm | Stage A | Stage B objective | Anti-collapse | Notes |
|---|---|---|---|---|
| J1 | none (or masked-pretrained) | mask-free k∈{1,2} dense | SIGReg | primary |
| J2 | block-masked pretraining | same | SIGReg | tests two-step learning |
| J3 | — | same | EMA | stability comparison |
| J4 | — | same | codebook | fallback / extra signals |
| JT | — | same, transformer encoder | SIGReg | capacity comparison |
| B0 | — | (old CARLA pipeline, frozen numbers) | margins | reference only |

Corpus axis on top: `{joint, single}` × above arms (joint exploratory only).

## 3. File layout (agreed)

```
models/
  __init__.py          # architecture registry
  resnet_ts.py         # renamed from resent_time.py (typo fix; imports updated)
  convolutions.py      # UNTOUCHED (incl. known stride caveat, documented here)
  jepa_pyramid.py      # JEPAPyramid (built; StridedConvBlock, CausalTCNPredictor)
  jepa_transformer.py  # transformer arm (kept per owner)
  jepa_core.py         # JEPAModel facade: encode/predict/compute_loss/score
  ema.py               # EMA target-model wrapper
  codebook.py          # soft-codebook module
losses/
  sigreg.py            # SIGReg loss class
  jepa_losses.py       # JEPA criterion class(es) (dense pred + λ·SIGReg [+ codebook])
data/
  jepa_dataset.py      # NEW JEPADataset (plain windows + optional preceding pair +
                       #   injected-anomaly probe view). Existing classes KEPT INTACT,
                       #   UNUSED: AugmentedDataset, DynamicNeighbors, ContrustiveDataset,
                       #   NeighborsDataset (owner decision D18).
utils/
  masking.py           # MaskingCollator (contiguous blocks, V-JEPA sampler mechanics in 1D)
  scoring.py           # Scorer + Calibrator (sub-window scores → per-timestep arrays → thresholds)
  trainer.py           # Trainer class (AMP, checkpoint/resume via existing format)
carla_jepa.py          # entry: --config_env --config_exp --fname --version, stage dispatch
configs/jepa/*.yml     # one YAML per arm; env.yml unchanged
REFERENCE_NUMBERS.md   # published SMD/PSM numbers, provenance-flagged
```

## 4. Deletion manifest (approved, D18-modified)

Delete once `carla_jepa.py` runs end-to-end:
`PretextLoss`, `ClassificationLoss`, `ClassificationLossE2E`, TCLoss wiring in `get_criterion`, `carla_pretext.py`, `carla_classification.py`, `experiments.py`, `configs/pretext/**`, `configs/classification/**`, `contrastive_evaluate`, `SilhouetteScore`, clustering-selection logic in checkpoint paths.

Keep untouched: `metrics/**`, `utils/utils.py::Logger`, `GradientMonitor`, `clean_checkpoint`, collate, SMD/PSM loaders, `SubAnomaly`/augmentations, env.yml, **and all four existing dataset classes** (unused but preserved per owner).

## 5. Build order

1. Foundation: registry `__init__`, `resnet_ts` rename, `ema.py`, `sigreg.py`, `jepa_losses.py`, `masking.py`, `jepa_dataset.py`.
2. Training: `trainer.py` (AMP + resume), `carla_jepa.py` stages pretrain/adapt.
3. Scoring/eval: `scoring.py` Calibrator→Scorer→per-timestep arrays → rewritten `pr_evaluate_timeseries` path → `configs/jepa/*.yml` for J1–JT.
4. Cleanup + docs: execute deletion manifest, write `REFERENCE_NUMBERS.md`, smoke run machine-1-1 (J1, single-machine), verify TensorBoard + metrics output end-to-end.

## 6. Risks carried forward

| Risk | Mitigation |
|---|---|
| Prediction-error-alone weakness (chance-level under fair protocols in literature) | ≥2 fused scores; no-training baseline reported alongside; fair single-dataset protocol only |
| Continuous self-distillation instability on TS | SIGReg primary; EMA/codebook arms isolate failure source |
| BatchNorm-inflated val loss (LeWM repro) | val loss computed in eval mode; prefer GELU/norm-light blocks |
| Published-number conventions differ wildly | provenance flags in `REFERENCE_NUMBERS.md`; report honest + PA separately (D4) |
| Overlapping windows (stride 5 < W 256) double-count points | scores aggregated across covering windows before thresholding (checking-window accumulation already handles detection side) |
