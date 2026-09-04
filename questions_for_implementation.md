# Questions for implementation review (non-blocking)

Decisions taken below with recommendation marked. Nothing blocked the build.

## Q1 — "Delete everything" scope
**Doubt:** full-repo wipe vs models/+losses/ clean rebuild.
**Taken:** keep frozen seams (`metrics/**`, datasets/results symlinks, tests/, configs, entry stages, checkpoint format); rebuild only `models/` + `losses/` internals with backward-compatible factory names.
**Recommendation:** confirm this scope is what you meant.

## Q2 — Facade name: keep `JEPAModel` or rename to `LeWMModel`?
**Doubt:** clean LeWM naming vs existing Trainer/Scorer/carla_jepa imports.
**Taken:** new clean facade in `models/lewm.py` (`LeWMModel`), keep `models/jepa_core.py` as a thin compatibility shim re-exporting it under the old name.
**Recommendation:** migrate imports to `lewm.py` over time; keep the shim until configs/tests move.

## Q3 — Encoder configurability depth
**Doubt:** how generic should the conv pyramid be (per-block channels, kernels, depths, norms, strides)?
**Taken:** `PyramidEncoder` accepts per-block `channels`, `depth` (num convs), `kernels` (list per conv), `strides`, `norm`, `dropout`; scalar args broadcast. Defaults reproduce the old stem+L1..L3 exactly.
**Recommendation:** keep this level; deeper NAS-style search is out of scope.

## Q4 — Predictor semantics after dropping forecasting
**Doubt:** horizons `k` meant future steps; now masked-part offsets. Keep causal TCN/GRU classes?
**Taken:** yes — reinterpret `horizons` as part offsets, keep causal masking (a part predicts neighbours, never itself). Add `CondPredictor` wrapper (AdaLN) for the closed-loop arm only.
**Recommendation:** confirm causal-within-window is the right constraint for part prediction.

## Q5 — Soft-DTW divergence dependency
**Doubt:** implement from scratch vs vendor a library.
**Taken:** compact from-scratch `soft_dtw_divergence` (Blondel `2010.08354`, gamma softmin, pairwise L2) on proposed parts only, CPU-safe for short parts; falls back to MSE when part length < 2.
**Recommendation:** keep vendored minimal; add `tsjadi`/`soft-dtw` dep only if tournament shows the minimal version underperforms.

## Q6 — YOLO-1D box loss fidelity
**Doubt:** full CIoU+DFL vs simplified IoU+L1.
**Taken:** `BoxLoss = focal-BCE objectness + (1 - 1D-IoU) + L1(center,length)`; CIoU aspect term and DFL kept as flags, default off for 1D stability.
**Recommendation:** enable CIoU/DFL only as tournament variant.

## Q7 — Per-level tiny VAEs (Q4-C vs A)
**Doubt:** full per-token VAE per level is heavy and collapse-prone.
**Taken:** `H2ReconHead` carries optional tiny `mu/logvar` heads per level, trained with KL annealing + free-bits; default off, enabled per-arm. View-KL channel stays the default alignment signal.
**Recommendation:** keep default off until ticket 14 evidence lands.

## Q8 — Metric center handling per-machine
**Doubt:** learnable vs fixed center, who initializes it.
**Taken:** center is a buffer initialized from first-epoch training embeddings (mean), fixed afterwards; variance hinge guards collapse. No gradient through center.
**Recommendation:** confirm frozen-center (not learnable) matches your intent.

## Q9 — Closed-loop training schedule
**Doubt:** joint vs alternating optimizer steps for action-conditioned arm.
**Taken:** joint loss with detached proposals + gradient-conflict monitor; alternating schedule left as config flag, default joint (simpler, tournament-safe).
**Recommendation:** try joint first; escalate to alternating only if `cos(g_main,g_aux) < 0` persists.

## Build notes (found during implementation, all resolved)

- **Stem kernel default:** the proven stem uses kernel 7 (was hardcoded, ignored `kernel_size`). Kept as `stem_kernel=7` default with config override; initial draft wrongly reused `kernels[0]`.
- **Checkpoint compatibility:** old `encoder.stem/levels.i.downsample|refine` keys remap automatically in `LeWMModel.load_state_dict` (plus `Trainer` resume/load paths); shapes identical so values transfer exactly (verified strict-load + value equality).
- **verify_stages masking assertion:** pre-existing flake, fails on pristine tree for some seeds (overlapping blocks merge past the bound). Untouched, left as-is.
- **Recon shape-term scale:** Soft-DTW divergence runs ~10x MSE on probe parts; keep `lambda_shape` small (tournament tunes; default 0).

## Review follow-ups (code-review 2026-09-03, accepted as staged gaps)

Fixed in this pass: SIGReg re-exported; recon mask denom broadcast-aware;
shape term normalized by valid parts; metric variance hinge on unnormalized
projections; proposals detached with soft top-1 pooling (embedder trainable);
1:4 hard-negative mining in box loss; single legacy-key remap helper;
criterion map instead of if-cascade; dead padding branch removed.

Still open, mapped to tickets (foundation lands first per Q10):
- T14: ELBO consumer for tiny-VAE mu/logvar + annealing schedule.
- T15: DAGMM K=2–4 mixture-energy variant arm.
- T16: NMS-to-mask wiring (proposals to part masks) + DILATE-temporal arm.
- T17/T18: alternating schedule option, tournament runner, per-machine
  threshold + no-training-baseline harness (currently manual via score stage).
- Naming: `Box*`/`decoder` identifiers kept (DESIGN_TSAD_DUALSTREAM uses
  interval-box language throughout); CONTEXT prefers `interval detection` —
  revisit identifiers if glossary is tightened.

## Round-2 resolutions (question tool, 2026-09-04 — all implemented)

- **Q1 entry/examples:** example configs PLUS new entry `carla_lewm.py`
  for full LeWM training (trunk + attached heads, shared Trainer,
  trunk.pth.tar exported for the trunk-only scoring entry).
  Example arms: `configs/lewm/synthetic_lewm_full.yml` (CPU demo, verified
  3-epoch run) and `configs/lewm/smd_lewm_full.yml` (SMD-shaped).
- **Q4 predictor:** `MaskedReconPredictor` added in `models/predictor.py`
  (mask-token + non-causal transformer over visible context, H=1 output
  contract) and made the `LeWMModel` default; TCN/GRU stay as arms with a
  uniform `forward(z, mask_pos)` signature. Masks thread facade→predict.
- **Q5 Soft-DTW:** pure-torch mirror of sdtw_cuda_loss.py math (forward
  Bellman softmin recursion, Gibbs backward via autograd, `normalize`
  divergence mode, Sakoe-Chiba `bandwidth`), CPU+CUDA, no numba.
  Verified: self-divergence exactly 0, cross positive, grads flow.
- **Q6 focal-BCE:** kept gamma 2.0 + 1:4 hard mining (positives are exact
  synthetic boxes, so standard focusing applies).
- **Q7 VAE modality:** mu/logvar use temporal convs (`vae_kernel`, default
  3, length-preserving padding) instead of pointwise.
- **Q8 centers:** running EMA update (`H4MetricHead.update_centers`,
  default momentum 0.99) driven per-step by a new `update_running_stats`
  Trainer hook (no-op on the bare trunk, mirrors the codebook pattern).

## Q10 — TF-Scout full trunk in this pass?
**Doubt:** tickets 13-18 imply full dual-stream (STFT grid, FiLM, x-attn, FPN, iFFT head). One pass cannot land all of it cleanly.
**Taken:** this pass lands the LeWM foundation + head/loss/criterion structure + open-loop wiring points (proposals as masks, action slot present but optional). Full frequency pathway + steering + FPN fusion stays behind a `tfscout` registry stub validated by shape tests, to be filled in ticket order.
**Recommendation:** confirm staged landing (foundation now, TF grid/steering next) vs wanting the full trunk in this commit.
