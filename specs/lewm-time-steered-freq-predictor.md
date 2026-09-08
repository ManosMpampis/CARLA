---
title: LeWM time-steered frequency predictor (single-scale ResNet encoder)
labels: [ready-for-agent]
created: 2026-09-08
branch: jepa
spec-source: DESIGN_TSAD_DUALSTREAM.md, specs/tf-scout-dualstream-losses.md, models/lewm_true.py, user-steering-proposal-2026-09-08
---

# SPEC: LeWM time-steered frequency predictor

## Problem Statement

`models/lewm_true.py` is not a true LeWM: it encodes `mask(X)` and `X` in two
passes with gradients on both, then feeds `cat[z_ctx, z_tgt]` into a disposable
projector for SIGReg (`lewm_true.py:150-155`). The predictor therefore sees
target-derived information through the pooled projection, and the action is a
post-hoc descriptor rather than the sole bridge between known and unknown states.

Separately, SMD favours small windows and small models, but small windows risk
sitting fully inside an anomaly (no normal context, no surprise) while `W=512`
degrades quality. The fix is a large parameterized window plus a time-only scout
that steers a frequency-only predictor to the suspect sub-window.

## Solution

Build a single-scale LeWM with fixed terminology:

- `ResNet Encoder`: time-domain only, whole world-state representation.
- `Convolution time auxiliary`: time-domain only, localizes the injected mask
  and steers the predictor. Never receives the action.
- `Predictor = frequency module only`: `FFT/STFT -> YOLO26-neck -> iFFT/iSTFT`,
  the sole action-conditioned path.

Training pair from `data/augment.py:SubAnomaly`:

```text
X_inj, mask = SubAnomaly(X_clean)          # mask: (B, W) binary
Z_whole  = Encoder(X_clean)                # (B, D, W)
Z_inj    = Encoder(X_inj)                  # (B, D, W), same weights
Time_mask = Auxiliary(Z_inj)               # (B, W) logits
Z'_whole  = Predictor(Z_inj, mask)         # (B, D, W)
Loss1 = MSE(sigmoid(Time_mask), mask)
Loss2 = KL | MSE | L1  (Z'_whole, Z_whole), default KL, config-selected
Loss  = w_pred*Loss2 + w_aux*Loss1 + lam*SIGReg(Z_whole) + l_*SIGReg(Z'_whole)
```

No `stop-grad` anywhere, no EMA teacher, no pooled projector. Both SIGReg terms
use the existing `losses/sigreg.py:SIGReg` statistic on separate branches.
Inference derives the action from the auxiliary:
`M_hat=Auxiliary(Encoder(X))`, `Z'=Predictor(Z, M_hat)`,
score `mean_d|Z'-Z|` per timestep.

## User Stories

1. As a researcher, I want the encoder to be a plain time-domain ResNet so that
   the world-state representation has no frequency leakage.
2. As a researcher, I want the auxiliary to see only `Encoder(X_inj)` so that
   it cannot trivially copy the action input.
3. As a researcher, I want the predictor to be frequency-only with the mask fed
   only at its stem so that LeWM action semantics hold.
4. As a researcher, I want `Loss1=MSE(Time_mask, mask)` so localization is
   supervised by the injected span.
5. As a researcher, I want `Loss2` selectable as `kl|mse|l1` with `kl` default
   so that latent reconstruction geometry is a config choice.
6. As a researcher, I want two separate SIGReg terms and no stop-grad so that
   collapse is prevented without a teacher.
7. As a researcher, I want single-scale latents so that scoring needs no pyramid
   fusion.
8. As a researcher, I want `W`, STFT params, and all channel widths parameterized
   so that 128/256/512 sweeps are YAML-only.
9. As a researcher, I want time-varying FiLM steering so that suspect times get
   different frequency treatment.
10. As an operator, I want inference to use `M_hat` as the action so that no
    ground-truth mask is needed at test time.
11. As a maintainer, I want registry + YAML arms, existing checkpoint/resume
    format, AMP-train/fp32-score, and the frozen metrics stack untouched.

## Implementation Decisions

- **Encoder (default, all parameterized):** 2 ResNet blocks; each block 3 main
  convs `k=[7,5,3]` stride 1 + 1 residual conv `k=5` stride 1; `Norm=GELU?`
  follow `models/blocks.py:ConvBlock1d` convention (`BatchNorm+GELU+Dropout`).
  `enc_channels=[32,64]` default (`in -> 32 -> D=64`); output `(B,D,W)`,
  `level_names=["L0"]`, `level_strides=[1]` to keep Trainer/criterion/scorer
  dict contracts with a single key.
- **Auxiliary:** 3-conv stack `k=[7,5,3]`, `aux_channels=[32,32,32]` default,
  stride 1, length-preserving padding; two 1x1 heads: (a) mask logits `(B,W)`,
  (b) FiLM features `(B,Da,W)` tapped before the mask head.
- **Mask source:** new collator `InputSubAnomalyCollator` wrapping
  `SubAnomaly.__call__`; emits `X_inj (B,C,W)` + binary temporal mask `(B,W)`
  (`1` on `[start,end)`, shared over perturbed channels). `SubAnomaly` itself
  keeps its distributions (`data/augment.py:94-186`); only the mask return is new.
  Dataset still yields `X_clean`; collator builds the pair. Validation collator
  identical (masks present, still train-distribution only).
- **Predictor stem + action:** `mask (B,1,W)` concatenated to `Z_inj (B,D,W)`
  -> `k=7` stem conv to `stem_channels` (default 64). No other action path.
- **Frequency body:** `STFT(n_fft=64, hop=16, win=64)` per stem channel
  (all parameterized) -> complex `(B,S,T',F)` -> stack `[real, imag]` ->
  `(B,2S,T',F)`; YOLO26-neck: 3 scales, lateral 1x1 to `neck_widths=[64,64,64]`
  (parameterized), top-down nearest-upsample+**concat**+conv, bottom-up
  stride-2-conv+**concat**+conv (user sketch); final 1x1 back to `2S` channels ->
  complex -> `iSTFT(length=W)` -> 1x1 proj to `D` -> `Z'_whole (B,D,W)`.
  `FFT/iFFT` (global RFFT) is a config ablation; STFT is the default because
  global spectra have no time axis for time-varying FiLM.
- **Steering (time-varying FiLM, default and only mode for v1):** per neck level
  with time resolution `T'_l`: `interp(aux_feat, T'_l)` -> per-timestep MLP ->
  `gamma,beta (B,Df,T'_l,1)` broadcast over `F`:
  `h=(1+gamma)*h+beta`. Injected at every neck level input. Global (pooled) and
  full-TF-cell `(T',F)`-varying FiLM are explicitly out of scope for v1.
- **Loss2 definitions (config `loss2_kind: kl|mse|l1`, default `kl`):**
  - `mse`: `((Z'-Z)**2).mean()` over `(B,D,W)`.
  - `l1`: `|Z'-Z|.mean()`.
  - `kl` (default): symmetric channel-wise KL without detach. Per `(b,t)`:
    `P=softmax(Z'/tau)`, `Q=softmax(Z/tau)` over `D`, `tau` parameterized
    (default 1.0); `0.5*(KL(P||Q)+KL(Q||P)).mean()` over `(B,W)`. Unlike
    `losses/alignment.py:ViewKLLoss`, neither side is detached per the
    no-stop-grad constraint.
  - `Loss1`: `MSE(sigmoid(Time_mask), mask.float()).mean()` over `(B,W)`.
  - Total: `w_pred*Loss2 + w_aux*Loss1 + lam*SIGReg(Z) + l_*SIGReg(Z')`
    with `w_pred=1.0, w_aux=1.0, lam=0.1, l_=0.1` defaults; criterion returns
    `loss` plus `pred_loss` (Loss2, the Trainer checkpoint-selection key),
    `aux_loss`, `sigreg`, `sigreg_tgt` for TensorBoard.
- **Model/criterion wiring:** new `SteeredFreqLeWMModel`
  (`encode/predict/score` methods, `target_encoder=None`, `anti_collapse=sigreg`)
  plus `SteeredLeWMLoss`; register backbone/criterion names
  (`steered_resnet`, `steered_lewm`) in `models/__init__.py` and
  `utils/common_config.py:CRITERION_BUILDERS`. Stages `pretrain/adapt/score`
  reuse `carla_lewm_true.py` flow; adapt keeps `frozen|finetune`; resume format,
  seed default 4, `amp` bf16-train only, fp32 scoring, eval-mode validation stats.
- **Scoring (single-scale):** eval mode, fp32: `Z=Encoder(X)`,
  `M_hat=sigmoid(Auxiliary(Z))`, `Z'=Predictor(Z, M_hat)`,
  per-timestep `mean_d|Z'-Z| -> (B,W)`; overlapping windows aggregated
  cover-count-aware into `(scores, start_idxs, end_idxs)`; thresholds from
  clean-train quantiles only (`calibration_kwargs.quantile`, default 0.995);
  injected probes used only as held-out calibrator signal, never shaping the
  representation; metrics stack frozen, honest vs point-adjust split kept.
- **Budgets:** order 1e5-1e6 params, single-GPU training, CPU-plausible scoring;
  `W` sweep 128/256/512 is expected (D=64, S=64 defaults must fit 512).

## Testing Decisions

- Repo convention only: plain-python assertion scripts under `tests/`
  (no pytest), plus YAML smoke chain on `machine-1-1.txt`.
- New `tests/verify_steered_lewm.py`: synthetic tiny config exercises
  pretrain->adapt->score; asserts (a) checkpoint resume bit-identical loss
  drop, (b) `Time_mask` shape `(B,W)` and `Loss1` finite, (c) each
  `loss2_kind` runs, (d) `score()` returns `(B,W)` fused with no NaN and
  `M_hat`-conditioned path differs from mask-zero path.
- Extend `tests/check_scorer_handoff.py` pattern: single-scale `(scores,
  start_idxs, end_idxs)` flows into frozen metrics dict unchanged;
  overlap aggregation covered-points-once check.
- Real-data smoke: `smd_pretrain/adapt/score_smoke`-style YAMLs for the new arm
  on `machine-1-1.txt` (CUDA, minutes); TensorBoard loss + graph present.

## Out of Scope

- EMA teacher, codebook, pooled projector, any stop-gradient, multi-scale
  pyramid fusion, transformer encoder/predictor arms, global or full-TF-cell
  FiLM, freq->time or bidirectional steering, global-RFFT-as-default,
  memory-bank heads, interval/YOLO output head, metrics-stack or
  datasets/results-symlink changes, joint-corpus headlines (per-machine only).

## Further Notes

- Source of neck pattern: user hand-sketch (FFT -> multi-node FPN/PAN with
  lateral concats, top-down upsample-add path, bottom-up downsample path,
  `out -> iFFT`); canonical refs FPN (Lin et al. 2017), YOLO neck, FiLM
  (Perez et al. 2018); repo pattern reuse: `PyramidEncoder` stride ladder
  collapsed to single stride-1, `Scorer` seam shape, `Calibrator` train-only
  thresholds, `SubAnomaly` probe machinery.
- Known hazard carried forward: prediction-error-only scoring has hit
  chance-level under fair protocols; the auxiliary mask channel plus
  no-training baseline comparison are mandatory companions to the fused score.
