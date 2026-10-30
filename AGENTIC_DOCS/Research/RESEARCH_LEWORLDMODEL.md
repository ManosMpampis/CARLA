# RESEARCH: LeWorldModel & the 2026 JEPA Ecosystem (update to RESEARCH_JEPA_WORLD_MODELS.md)

Date: 2026-08-23. Scope: research only; companion to `RESEARCH_JEPA_WORLD_MODELS.md` (LeCun 2022 framing, I-JEPA, V-JEPA/2/2-AC, SC-JEPA, T-SAR-JEPA, CF-JEPA, PULS are covered there and NOT re-verified here). Every factual claim carries a primary-source citation fetched this session unless flagged otherwise.

---

## 1. TL;DR

- **"LeWorldModel" is a real artifact**, not shorthand for LeCun's program: *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels* (Maes\*, Le Lidec\*, Scieur, **LeCun**, Balestriero; [arXiv:2603.19312](https://arxiv.org/abs/2603.19312); official code [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm), MIT).
- Its headline recipe: JEPA trained **end-to-end from pixels** with only two terms — next-embedding MSE prediction + **SIGReg** regularizer enforcing isotropic-Gaussian latents. **No stop-gradient, no EMA teacher, no pretrained encoder.** ~15M params, single GPU ([abstract](https://arxiv.org/abs/2603.19312)).
- Anti-collapse = SIGReg from LeJEPA ([arXiv:2511.08544](https://arxiv.org/abs/2511.08544)): project latents on M random unit directions, per-projection Epps–Pulley normality statistic (Cramér–Wold). Default M=1024, weight **λ=0.1**; #projections nearly irrelevant in ablations → **λ is the single effective hyperparameter** ([App. A](https://arxiv.org/html/2603.19312v3)).
- Anomaly-relevant result: LeWM's **Violation-of-Expectation surprise** (latent prediction discrepancy) spikes sharply on physical perturbations across 3 envs (p<0.01) — prediction-error-as-anomaly-signal works inside this framework ([§5.2](https://arxiv.org/html/2603.19312v3)).
- **Critical caveat:** an independent reproduction ([arXiv:2608.10145](https://arxiv.org/abs/2608.10145)) needed four undocumented conventions to converge; authors' checkpoint scores 84% under their config protocol but **14% under the paper-appendix protocol**; a BatchNorm layer inflated validation loss up to **300×**; one-step prediction accuracy does NOT predict long-horizon success. Any LeWM-derived design must re-verify rather than trust configs.
- V-JEPA 2.1 ([arXiv:2603.14482](https://arxiv.org/abs/2603.14482), released 2026-03-16) adds a **Dense Predictive Loss**: context (visible) tokens also carry explicit L1 targets with distance weighting λ_i = λ/√(d_min(i,M)) — making **per-token/per-sub-window prediction errors well-defined at inference**, which is exactly the enabler for sub-window anomaly scoring.
- **Sub-window TS localization gap:** no published work does explicit sub-window-level AD scoring inside a time-series window with a JEPA on standard MTS benchmarks. Closest evidence: SD-JEPA's subspace-decomposed progression coordinate beats scalar surprise for event localization (+0.18 pooled AUROC, [arXiv:2605.31111](https://arxiv.org/abs/2605.31111)); T-SAR-JEPA scores per acquisition; PULS-successor scores per video chunk. The niche is open.

---

## 2. LeWorldModel — verified specifics

### 2.1 Architecture and loss
| Item | Value | Source |
|---|---|---|
| Paper / code | arXiv:2603.19312 (v1 2026-03-13, v3 2026-06-03) / github.com/lucas-maes/le-wm | https://arxiv.org/abs/2603.19312 ; https://github.com/lucas-maes/le-wm |
| Loss | `pred_loss = MSE(emb[:,1:] − next_emb[:,:−1])` + step-wise `0.1·SIGReg(emb)` applied **per time-step over batch** (training pseudo-code embedded in paper Fig. 9) | https://arxiv.org/html/2603.19312v3 |
| Encoder | HF ViT-Tiny patch-14 (~5M); representation = **[CLS] → 1-layer MLP projector with BatchNorm** — explicitly required because final LayerNorm blocks optimizing Gaussianity | same |
| Predictor | ViT-S-size transformer (~10M ≈ **2× encoder params**), learned pos-emb, causal mask, history N=3 (N=1 TwoRoom); actions injected per-layer via zero-init AdaLN (DiT trick) | same |
| Data pipeline | frameskip 5 (consecutive actions grouped into action blocks), batch 128, sub-trajectories of 4 frames, 224² | same |
| Planning/MPC cost | CEM 300 samples, 10–30 iters, top-30 elites, horizon 5; cost = MSE between last predicted latent and goal latent (`criterion` method in jepa.py) | https://raw.githubusercontent.com/lucas-maes/le-wm/main/jepa.py (fetched) |

### 2.2 SIGReg mechanics (from paper App. A + LeJEPA)
- M unit directions sampled uniform on sphere; per-direction univariate Epps–Pulley statistic T = ∫ w(t)|φ_N(t;h)−φ₀(t)|² dt, Gaussian weighting w(t)=exp(−t²/2λ²); trapezoid quadrature, nodes uniform in **[0.2, 4]** (https://arxiv.org/html/2603.19312v3 App. A).
- Upstream theory: isotropic-Gaussian embedding distribution proven optimal for minimizing downstream prediction risk; SIGReg ≈50 lines; no stop-grad/teacher/schedulers; ViT-H/14 → 79% IN-1K linear probe (Balestriero & LeCun, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544), code github.com/rbalestr-lab/lejepa).

### 2.3 Independent reproduction red flags (must-read)
- Singh, *"The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom"* ([arXiv:2608.10145](https://arxiv.org/abs/2608.10145), 2026-08-10; code github.com/joyjeet-singh/tinylab): ~$25 compute reproduction succeeded only after four undocumented conventions (dense action gathering across frameskip block, programmatically-set action-encoder width, ImageNet normalization, action z-scoring) that "appear in no released configuration file"; following released configs alone → predictor never converges. Authors' own checkpoint: **84.0% under their config protocol vs 14.0% under the paper-appendix protocol**. A BatchNorm layer inflated reported validation loss up to **300×**. One-step prediction accuracy does not predict long-horizon planning success.
- Follow-up (*"The Objective Is the Bottleneck"*, [arXiv:2608.12959](https://arxiv.org/abs/2608.12959)): CEM's squared-latent-distance cost saturates beyond ~120 units; replacing only the planner objective lifts goals-reached 26%→98%.

### 2.4 LeWM ecosystem (all arXiv-verified this session)
| Work | arXiv | Relevance |
|---|---|---|
| Fast LeWorldModel | [2606.26217](https://arxiv.org/abs/2606.26217) | action-prefix parallel prediction instead of autoregressive rollout; lower latent-error growth |
| Hi-LeWM (hierarchical planning) | [2607.12547](https://arxiv.org/abs/2607.12547) | frozen low-level LeWM + latent-subgoal planner; WM@Booth 2026 |
| Temporally Centered SIGReg | [2607.26924](https://arxiv.org/abs/2607.26924) | apply SIGReg to temporally centered residuals, not raw latents (LIBERO 53.2→73.6%) |
| QQWorld | [2607.28415](https://arxiv.org/abs/2607.28415) | Epps–Pulley gradients vanish for tail samples → quantile-quantile matching instead |
| Sub-JEPA | [2605.09241](https://arxiv.org/abs/2605.09241) | enforce Gaussianity only in random subspaces (isotropic bias too strong) |
| AC-MTM ("No Gaussian Required") | [2608.17542](https://arxiv.org/abs/2608.17542) | anti-collapse from inverse-dynamics head, training-only; beats SIGReg on OGBench Visual Scene (80±2% vs 58±2%) |
| RC-aux | [2605.07278](https://arxiv.org/abs/2605.07278) | multi-horizon + reachability auxiliary objectives on LeWM |
| SCALE | [2608.16287](https://arxiv.org/abs/2608.16287) | calibrate LeWM latent geometry toward true state distances for better CEM costs |
| GC-IDM | [2605.08732](https://arxiv.org/abs/2605.08732) | amortize planning into goal-conditioned inverse dynamics (100–130× cheaper/decision) |
| TRM | [2605.22164](https://arxiv.org/abs/2605.22164) | post-hoc terminal-ranking head; TwoRoom 7→97% where raw latent planning fails |
| FF-JEPA | [2606.09311](https://arxiv.org/abs/2606.09311) | hierarchical action-free planner on frozen LeWM encoder+predictor |
| Causal-JEPA (ICML 2026) | [2602.11389](https://paperswithcode.co/paper/2602.11389) | object-centric masking/interventions; same author cluster as LeWM |

---

## 3. Official Meta releases since our last doc

### V-JEPA 2.1 (released 2026-03-16)
- Paper: *V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning* (Mur-Labadia, Muckley, Bar, Assran, Sinha, Rabbat, LeCun, Ballas, Bardes; FAIR+Univ. Zaragoza), [arXiv:2603.14482](https://arxiv.org/abs/2603.14482); repo changelog entry "V-JEPA 2.1 is released" at [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2); torch.hub `vjepa2_1_vit_{base,large,giant,gigantic}_384`.
- **Dense Predictive Loss** (§2.3): keep masked-token L1 loss ℒ_predict AND add ℒ_ctx on visible/context tokens: ℒ_ctx = (1/|C|)Σ_i λ_i‖P(E(x),Δ)_i − sg(Ē(y))_i‖₁ with **λ_i = λ/√(d_min(i,M))** (distance to nearest mask block). Fixed λ=0.5 trades segmentation vs classification (ADE20K 33.8 mIoU but SSv2 72.8→62.5); **warm-up over epochs 50–100 recovers both** (Table 2). Rationale: without it, context tokens degenerate into register-token-style global aggregators (§2.2).
- **Deep Self-Supervision**: predictor outputs predictions at 4 levels (concatenating outputs of 3 intermediate encoder blocks + final, MLP-fused); both losses at every level.
- Headline: NYUv2 depth 0.307 RMSE, ADE20K 47.9 mIoU, SSv2 77.7%, +20% real-Franka grasping success over V-JEPA 2-AC (abstract).
- No V-JEPA 3 exists (no trace, arXiv+web, 2026-08-23). "APIJEPA": zero arXiv hits — treat as nonexistent/unreleased. No official Meta time-series JEPA exists.

### Official-code implementation details worth stealing (raw files fetched today)
From https://raw.githubusercontent.com/facebookresearch/vjepa2/main/app/vjepa/train.py :
- Loss is a plain configurable L_p norm: `torch.mean(torch.abs(zij - hij) ** loss_exp) / loss_exp`; shipped configs use `loss_exp: 1.0` = pure **L1** (no smooth-L1, no cosine).
- Targets get `F.layer_norm(h)` applied **before** masking/loss (per-feature normalization trick).
- Target encoder = `copy.deepcopy(encoder)`, requires_grad=False; EMA schedule is linear interpolation `ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)` stepped per iteration via `k ← m·k + (1−m)·q`.
From https://raw.githubusercontent.com/facebookresearch/vjepa2/main/configs/train/vitl16/pretrain-256px-16f.yaml :
- `ema: [0.99925, 0.99925]` (**constant**, no ramp); lr 5.25e-4 (start 1e-4, warmup 40 ep), wd 0.04, bf16, seed 239; predictor `pred_depth=12, pred_embed_dim=384, pred_num_heads=12` (≈⅜ encoder width, ½ depth).
From https://raw.githubusercontent.com/facebookresearch/vjepa2/main/src/masks/multiseq_multiblock3d.py :
- Block-mask sampler: ONE block size per sample-seed (temporal/spatial scale ranges, aspect-ratio range), then `num_blocks` random translated copies; union = target mask; **encoder mask = complement**; `max_temporal_keep` zeroes context beyond first X tubelets (causality); options full_complement / inv_block. Shipped video config uses TWO groups: {num_blocks 8, spatial_scale [0.15,0.15]} + {num_blocks 2, spatial_scale [0.7,0.7]}, temporal_scale [1.0,1.0], aspect_ratio [0.75,1.5].
- Predictor-size rule-of-thumb across family: V-JEPA 2 ≈ ⅜ width / ½ depth; LeWM ≈ 2× encoder params; TS-JEPA = 1× encoder. Nobody ablates this axis rigorously except LeWM (App. G).
- I-JEPA exact mask numbers (context block 0.85–1.0 area; 4 targets at 0.15–0.20, AR 0.75–1.5) come from a secondary synthesis (themind.io blog) — not re-fetched from primary configs this session (flagged).

---

## 4. Time-series JEPAs since mid-2025 (beyond SC-/T-SAR-/CF-JEPA)

Verified via arXiv API `all:"JEPA" AND all:"time series"` (21 results, 2026-08-23):

| Paper (arXiv) | Encoder / masking | Collapse prevention | Predictor | AD relevance |
|---|---|---|---|---|
| **TS-JEPA** "Joint Embeddings Go Temporal" [2509.25449](https://arxiv.org/abs/2509.25449) (NeurIPS'24 WS) | Transformer d=128; non-overlapping patches → 1D-CNN embed + sin-cos pos-enc; 10 patches/series; **uniform random point/patch masking 70%** | **EMA target encoder m=0.998 constant** | Transformer, same size as encoder; L1 avg over masked patches | None (classification/forecasting only). Code: github.com/Sennadir/TS_JEPA |
| **LeNEPA** [2607.00958](https://arxiv.org/abs/2607.00958) (KDD MILETS 2026; Chemeris, Jin, **Balestriero**) | causal backbone; **next-latent-token objective, no augmentations, mask-free** | **SIGReg** (replaces stop-gradient/EMA) | lightweight; predictive loss computed in a projected space discarded at eval | frozen-probe classification (PTB-XL/Diag/UCR-128). Code: github.com/langotime/lenepa-milets-2026 |
| **HEPA** [2605.11130](https://arxiv.org/abs/2605.11130) (Spotlight FMSD@ICML 2026) | causal Transformer; **horizon-conditioned predictor** forecasts future representations (no reconstruction) | JEPA pretraining then freeze encoder, finetune predictor into monotonic survival-CDF head | reused for event probability over horizons | **Event/anomaly detection**: water contamination, cyberattacks, volatility regimes, 11 domains; beats PatchTST/iTransformer/MAE/Chronos-2 on ≥10/14 benchmarks. Code: github.com/Forgis-Labs/HEPA |
| **Phys-JEPA** [2606.16076](https://arxiv.org/abs/2606.16076) | states decomposed physical + residual | predictive + physics-consistency objectives | latent transition predictor | forecasting (Jena/Traffic/Electricity) |
| **CHARM** [2605.31580](https://arxiv.org/abs/2605.31580) (ICML'26 version; earlier [2505.14543](https://arxiv.org/abs/2505.14543)) | channel-order-equivariant Transformer, 7M params; latent prediction between views | JEPA + stability loss aligning predicted vs target embeddings at **pointwise, channel-mean, and global-mean levels** | JEPA predictor | AD among tasks via linear probe |
| ER-JEPA [2607.01145](https://arxiv.org/abs/2607.01145) | hierarchical H-JEPA (ViT intervals → second JEPA over interval reps) | hierarchy | per-stage predictors | ECG downstream SOTA |
| CGM-JEPA [2605.00933](https://arxiv.org/abs/2605.00933) | masked latent representation prediction (+cross-view X-variant) | predictive SSL | masked-latent predictor | subphenotype AUROC |
| SPLICE [2605.00126](https://arxiv.org/abs/2605.00126) | daily load segments → 64-dim pooled latents | JEPA pretraining + conformal envelopes (ACI) | conditional latent bridge (flow matching) | imputation; relevant for conformal calibration of latent errors |
| Cardiac action-conditioned JEPA [2604.22618](https://arxiv.org/abs/2604.22618) | disease onset as action/transition vector on latent state | **SIGReg** (adapting LeJEPA to physiological TS) | action-conditioned predictor | triage +0.05 AUROC low-label |
| Light-curve JEPA [2606.28446](https://arxiv.org/abs/2606.28446) | uncertainty-aware tokenization, irregular sampling | **LeJEPA regularization** | self-distillation heads | photometric zero-point drift detection |
| Koopman-JEPA theory [2511.09783](https://arxiv.org/abs/2511.09783) | idealized JEPA | constraining linear predictor near identity forces encoder onto Koopman eigenfunctions (= dynamical-regime indicators) | linear/near-identity | theory; interpretable regimes hint |
| ECG SSL systematic study [2605.12241](https://arxiv.org/abs/2605.12241) | SSM/Transformer/CNN ≤11M samples | five objectives incl. JEPA | — | finding: **CPC slightly ahead of JEPA** for transfer; SSMs beat transformers/CNNs |

Not verified further: "MTS-JEPA" has **zero primary records** (likely aggregator alias for SC-JEPA's multi-resolution objective); Brain-JEPA [2409.19407], TimeCapsule [2504.12721], LaT-PFN [2405.10093], Girgis TS-JEPA [2406.04853] exist but were not deep-read this session.

---

## 5. Sub-window / per-timestep anomaly localization from JEPA latents

**Bottom line: NO published work does explicit sub-window-level AD scoring inside a time-series window with a JEPA on standard MTS benchmarks (searched 2026-08-23). The niche is open.** Closest verified evidence:

1. **SD-JEPA — Subspace-Decomposed JEPAs** ([arXiv:2605.31111](https://arxiv.org/abs/2605.31111)): carves LeWM latent into progression subspace (cosine-margin triplet loss) + content subspace (SIGReg). Their 1-D angular progression coordinate |Δθ_t| localizes semantic events within episodes better than scalar surprise: **up to +0.18 pooled AUROC at ±1-step tolerance, 97.5% per-episode win rate**; quote: "separating the moment of surprise from its meaning in a way that prediction-error scalars cannot."
2. **T-SAR-JEPA** exact scoring formula: per-acquisition a_i = ‖ẑ_i − z_i‖₂ (prediction vs observed embedding; K=7 context, 4-layer causal temporal transformer) — https://arxiv.org/html/2606.05700v1 ; code github.com/TerraLatent/t-sar-jepa.
3. **V-JEPA 2.1 Dense Predictive Loss**: because *every* token carries an explicit L1 target, **per-token errors are well-defined and meaningful** at inference — the key architectural enabler for emitting per-sub-window scores instead of one window-level scalar (https://arxiv.org/html/2603.14482v2 §2.2–2.3).
4. **Kepler-Encoder-v0.1** ([arXiv:2607.13522](https://arxiv.org/abs/2607.13522)): robot multimodal LeJEPA/SIGReg encoder whose cross-modal prediction error is a training-free invalid-state monitor (**AUROC 0.90** out-of-range states) — per-single-timestep scoring without labels.
5. Zero-label driving complexity ([arXiv:2606.28383](https://arxiv.org/abs/2606.28383)): temporal prediction error as zero-shot score; AP 0.512 vs 0.436 chance — feasible but weak alone.
6. Latent Clarity / PULS successor ([arXiv:2607.03558](https://arxiv.org/abs/2607.03558)): chunk-level AUROC 0.8994 UCF-Crime, 0.8162 XD-Violence; L1-surprise gate on anticipated-vs-observed latent discrepancy. Chunk = video-domain sub-window granularity.
7. LeWM VoE curves: per-frame surprise localizes teleportation timestep precisely (App. F.3, https://arxiv.org/html/2603.19312v3).
8. CHARM pointwise alignment level implies per-point embedding discrepancies available at scoring time ([2605.31580](https://arxiv.org/abs/2605.31580)).

Design implication supported by these sources: combine (a) V-JEPA 2.1-style all-token supervision so each sub-window has a calibrated latent target, (b) SD-JEPA-style dedicated subspace(s) so representation change is measurable against stable geometry, (c) SIGReg or codebook anti-collapse (SC-JEPA documented continuous-self-distillation instability on TS). No paper combines these yet for MTS AD.

---

## 6. Limitations

- HEPA, ER-JEPA, Phys-JEPA, CGM-JEPA full texts not fetched — collapse-prevention/masking specifics inferred from abstracts only.
- I-JEPA exact EMA ramp values not re-fetched from primary configs; scheduler mechanism is primary-verified in vjepa2 train.py.
- V-JEPA 2.1 dates conflict across artifacts (repo release 2026-03-16; arXiv v1 2026-03-17/18; v2 HTML footer Aug 11, 2026) — March 2026 treated as release.
- "Interpreting Physics in Video World Models" seen only as a title on Meta's publications page; not fetched.
- Google Scholar inaccessible; discovery via arXiv API + web search + GitHub/HF. A few alphaxiv/paperswithcode mirrors used only to locate primary IDs.
- Three incidental summaries (AIF/free-energy JEPA theory; depth-prior SIGReg agricultural robot; BioM-JEPA single-cell) encountered but NOT independently verified — excluded.

## 7. Sources (accessed 2026-08-23)

Primary papers (all `https://arxiv.org/abs/<id>`): 2603.19312 (LeWM) · 2511.08544 (LeJEPA) · 2608.10145 · 2608.12959 (reproductions) · 2606.26217 · 2607.12547 · 2607.26924 · 2607.28415 · 2605.09241 · 2608.17542 · 2605.07278 · 2608.16287 · 2605.08732 · 2605.22164 · 2602.11389 · 2606.09311 (LeWM ecosystem) · 2603.14482 (V-JEPA 2.1) · 2509.25449 (TS-JEPA) · 2607.01145 (ER-JEPA) · 2607.00958 (LeNEPA) · 2606.16076 (Phys-JEPA) · 2605.31580 · 2505.14543 (CHARM) · 2605.11130 (HEPA) · 2605.00933 (CGM-JEPA) · 2605.00126 (SPLICE) · 2604.22618 (cardiac AC-JEPA) · 2606.28446 (light-curve) · 2511.09783 (Koopman-JEPA) · 2605.12241 (ECG study) · 2605.31111 (SD-JEPA) · 2607.13522 (Kepler) · 2606.28383 · 2607.03558 · 2606.05700 (T-SAR-JEPA).

Code (raw files actually fetched today):
- https://raw.githubusercontent.com/lucas-maes/le-wm/main/jepa.py
- https://raw.githubusercontent.com/facebookresearch/vjepa2/main/app/vjepa/train.py
- https://raw.githubusercontent.com/facebookresearch/vjepa2/main/src/masks/multiseq_multiblock3d.py
- https://raw.githubusercontent.com/facebookresearch/vjepa2/main/configs/train/vitl16/pretrain-256px-16f.yaml

Repos/pages: github.com/lucas-maes/le-wm · le-wm.github.io · huggingface.co/collections/quentinll/lewm · github.com/facebookresearch/vjepa2 · ai.meta.com/research/vjepa · github.com/rbalestr-lab/lejepa · github.com/Forgis-Labs/HEPA · github.com/Sennadir/TS_JEPA · github.com/TerraLatent/t-sar-jepa · github.com/joyjeet-singh/tinylab.

Secondary (flagged inline): themind.io JEPA history (I-JEPA mask numbers) · emergentmind.com topics page ("MTS-JEPA" alias, no primary source found).
