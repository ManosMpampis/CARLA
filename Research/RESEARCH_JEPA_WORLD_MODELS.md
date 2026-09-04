# RESEARCH: JEPA & LeCun World Models for Anomaly Detection (with Time-Series Focus)

Date: 2026-08-23. Scope: research only. All factual claims carry primary-source citations; anything not traceable to a primary source is labeled **Inference:** or **Speculative:** or excluded (see §6/§7).
Grounding for §5: local `AGENTS.md` and `WORKFLOW_PRETEXT.md`.

---

## 1. TL;DR

- JEPA = two encoders + a predictor operating **entirely in latent space**; LeCun's text defines "The energy is the prediction error" ([LeCun 2022, Fig. 12 caption](https://openreview.net/forum?id=BZ5a1r-kVsf)), making JEPA literally an energy machine whose energy surface is usable as an anomaly score.
- Collapse prevention in practice = **EMA target encoder + stop-gradient + masked latent prediction** (verified in [I-JEPA](https://arxiv.org/abs/2301.08243) and [V-JEPA Eq. (1)](https://arxiv.org/abs/2404.08471)); LeCun frames the alternatives as contrastive vs. regularized methods with four informational criteria ([LeCun 2022](https://openreview.net/forum?id=BZ5a1r-kVsf)).
- Verified prior work applies JEPA-family models to anomaly detection in **video** ([PULS](https://arxiv.org/abs/2607.03558); cautionary counterexample [Asleep at the Wheel](https://arxiv.org/abs/2608.01336)) and in **time series** ([SC-JEPA](https://arxiv.org/abs/2602.04643), [T-SAR-JEPA](https://arxiv.org/abs/2606.05700), [automotive monitoring](https://arxiv.org/abs/2602.09985)). No verified JEPA-for-industrial-image-defect work surfaced on arXiv (negative search, §4.6).
- For THIS repo (SMD/PSM pretext→classification): a JEPA objective maps naturally onto the existing three-view pipeline — the "near" view is *usually the preceding raw window* (per `WORKFLOW_PRETEXT.md`), i.e., a ready-made (context, future) pair for latent window-prediction. All integration details are design inference (§5).

---

## 2. The JEPA family and LeCun's world-model framework

### 2.1 Energy-based framing

| Claim | Primary evidence |
|---|---|
| SSL objectives can be cast as Energy-Based Models (EBMs); an EBM scores compatibility of x, y through scalar energy F(x,y) | [LeCun 2022, §4.3 text](https://web.archive.org/web/2023id_/https://openreview.net/pdf?id=BZ5a1r-kVsf) ("A general formulation can be done with the framework of Energy-Based Models") |
| Latent-variable EBMs infer z by minimizing E(x,y,z); z carries information about y not extractable from x (e.g., did the car turn left or right?) | [LeCun 2022, §4.3](https://openreview.net/forum?id=BZ5a1r-kVsf) ("the inference procedure of the EBM finds a value of the latent variable z that minimizes the energy") |
| JEPA: branch encoders produce s_x, s_y; a predictor outputs ŝ_y from s_x (optionally with latent z); "The energy is the prediction error" | [LeCun 2022, Fig. 12 caption](https://openreview.net/pdf?id=BZ5a1r-kVsf) |
| Main advantage: predictions in representation space, avoiding prediction of every detail of y; encoders eliminate irrelevant/unpredictable details | [LeCun 2022, §4.4–4.6](https://openreview.net/pdf?id=BZ5a1r-kVsf) |

### 2.2 Collapse prevention (what the literature specifies)

| Mechanism | Primary evidence |
|---|---|
| A joint-embedding architecture "can collapse when the encoders ignore the inputs and produce constant and equal codes" | [LeCun 2022, Fig. 10 discussion](https://openreview.net/pdf?id=BZ5a1r-kVsf) |
| Two remedy families: contrastive methods (push down training energies, pull up contrastive samples) vs. regularized/non-contrastive methods | [LeCun 2022, §4.3–4.5](https://openreview.net/pdf?id=BZ5a1r-kVsf) |
| Four non-contrastive criteria: maximize information content of s_x and of s_y; make s_y predictable from s_x; minimize latent-variable information | [LeCun 2022, §4.5, Fig. 13](https://openreview.net/pdf?id=BZ5a1r-kVsf) |
| I-JEPA instantiation: target-encoder weights updated as an **exponential moving average** of the context encoder; loss = **average L2 distance** between predicted patch-level and target patch-level representations; gradient flows only through context encoder + predictor (stop-grad via EMA targets) | [I-JEPA full text](https://arxiv.org/html/2301.08243v3) ("updated at each iteration via an exponential moving average…", "The loss is simply the average L2 distance…") |
| V-JEPA instantiation: explicit objective `min ‖P_φ(E_θ(x), Δy) − sg(Ē_θ(y))‖₁` with stop-gradient `sg`; "incorporating an exponential moving average … ensures that the predictor evolves faster than the encoder … thereby preventing collapse" | [V-JEPA full text, Eq. (1)](https://arxiv.org/html/2404.08471v1) |
| Masking matters: crucial to sample large-scale target blocks and a spatially distributed context block | [I-JEPA full text](https://arxiv.org/html/2301.08243v3) |

### 2.3 Concrete mechanisms (I-JEPA / V-JEPA / V-JEPA 2)

- **I-JEPA** ([arXiv:2301.08243](https://arxiv.org/abs/2301.08243), CVPR 2023): non-generative SSL; from one context block, predict representations of several target blocks of the *same image*; explicitly avoids hand-crafted augmentations and pixel-level fill-in; ViT-H/14 trained on ImageNet in <72 h on 16 A100s ([abstract](https://arxiv.org/abs/2301.08243)). Official repo describes the predictor as "a primitive (and restricted) world-model" modeling spatial uncertainty from partial context; repo was archived Aug 1, 2024 ([facebookresearch/ijepa README](https://github.com/facebookresearch/ijepa)).
- **V-JEPA** ([arXiv:2404.08471](https://arxiv.org/abs/2404.08471)): feature prediction as a stand-alone video objective — no pretrained image encoders, no text, no negatives, no reconstruction; trained on 2M public videos; evaluated with frozen backbones (ViT-H/16: 81.9% Kinetics-400, 72.2% SSv2, 77.9% ImageNet-1K) ([abstract](https://arxiv.org/abs/2404.08471)).
- **V-JEPA 2** ([arXiv:2506.09985](https://arxiv.org/abs/2506.09985), 11 Jun 2025): action-free JEPA pretraining on >1M hours internet video; 77.3 top-1 SSv2; 39.7 R@5 Epic-Kitchens-100 anticipation; **V-JEPA 2-AC** = latent *action-conditioned* world model post-trained with <62 h unlabeled Droid robot data, deployed zero-shot on Franka arms for pick-and-place with image goals, no task-specific training or reward ([abstract](https://arxiv.org/abs/2506.09985)). Official repo ships checkpoints/configs and an `energy_landscape_example.ipynb` "computing the energy landscape of the pretrained action-conditioned backbone" ([facebookresearch/vjepa2 README](https://github.com/facebookresearch/vjepa2)); V-JEPA 2.1 (released 2026-03-16 per repo changelog) adds a Dense Predictive Loss where all tokens contribute to the loss ([vjepa2 README](https://github.com/facebookresearch/vjepa2)).
- **Official framing of world-model roles** (Meta AI blog): world models need *Understanding*, *Predicting*, *Planning*; V-JEPA 2 = encoder + predictor built on a JEPA ([Meta AI V-JEPA 2 blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks); accessed via [Wayback snapshot 2025-06-12](https://web.archive.org/web/202506120249/https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) because ai.meta.com blocks scripted access). The I-JEPA blog calls I-JEPA "the first AI model based on Yann LeCun's vision," comparing "abstract representations of images (rather than comparing the pixels themselves)" ([Meta AI I-JEPA blog](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/), via [Wayback 2023-07-12](https://web.archive.org/web/20230712232736/https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)).

### 2.4 LeCun's autonomous-agent architecture (component roles)

From [LeCun 2022, §3, Fig. 2](https://openreview.net/pdf?id=BZ5a1r-kVsf):

| Module | Role (as specified) |
|---|---|
| Configurator | Executive control; configures perception, world model, cost, actor for the task at hand |
| Perception | Estimates current world state from sensors, hierarchically, multiple abstractions |
| World model | Predicts plausible future world states given imagined action sequences; latent variables parameterize prediction uncertainty; "a kind of 'simulator'" |
| Cost | Scalar "energy" = immutable intrinsic cost + trainable **critic** predicting future intrinsic-cost values |
| Actor | Proposes action sequences; optimizes via the world model + critic; emits first action of optimal sequence |
| Short-term memory | Tracks current/predicted states and costs |

Hierarchical variant: **H-JEPA** stacks JEPAs — JEPA-1 learns low-level representations with short-term predictions; JEPA-2 consumes those for higher-level, longer-term predictions; abstraction discards hard-to-predict detail ([LeCun 2022, §4.6, Fig. 15](https://openreview.net/pdf?id=BZ5a1r-kVsf)).

---

## 3. Why prediction-error-in-latent-space fits anomaly detection

**Energy argument (primary text).** In LeCun's formulation the JEPA energy *is* the prediction error D(s_y, ŝ_y) in representation space ([Fig. 12 caption](https://openreview.net/pdf?id=BZ5a1r-kVsf)), and EBM inference = finding y (or z) minimizing energy ([§4.3](https://openreview.net/pdf?id=BZ5a1r-kVsf)). Low energy ≈ "plausible continuation of the past"; high energy = input inconsistent with learned dependencies. LeCun explicitly ties surprise to world models in prose: agents "dismiss interpretations that are not consistent with their internal world model, and pay special attention as it may indicate a dangerous situation" ([§2.1](https://openreview.net/pdf?id=BZ5a1r-kVsf)). Meta's own V-JEPA 2 repo demonstrates operationalizing this: an official notebook sweeps candidate futures and visualizes the model's **energy landscape** over robot trajectories ([vjepa2 README](https://github.com/facebookresearch/vjepa2)). Published JEPA-adjacent detectors use exactly this quantity as the anomaly score (see §4).

**Conditions for latent prediction error to be a usable anomaly score** — each condition cited; the synthesis line is reasoning:

1. Train only on normal data, so the learned predictor covers the inlier manifold (standard AD protocol; e.g., [T-SAR-JEPA](https://arxiv.org/abs/2606.05700) trains self-supervised then scores against held-out events).
2. Targets must stay in latent space, not pixels, so unpredictable nuisance detail is discarded rather than dominating the score — the core JEPA claim ([LeCun 2022 §4.4](https://openreview.net/pdf?id=BZ5a1r-kVsf); [I-JEPA](https://arxiv.org/html/2301.08243v3)).
3. Collapse must be prevented, otherwise energies flatten and scores saturate (EMA+stop-grad+masking per [V-JEPA Eq. 1](https://arxiv.org/html/2404.08471v1); criteria list in [LeCun 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf)). SC-JEPA reports exactly this failure mode on time series: "directly applying continuous self-distillation to time-series data is often unstable and can lead to representation collapse" ([arXiv:2602.04643](https://arxiv.org/abs/2602.04643)).
4. Evaluation protocol must not confound novelty with domain shift — [Asleep at the Wheel](https://arxiv.org/abs/2608.01336) shows cross-dataset protocols silently reward domain separation while a fair single-dataset benchmark drives JEPA prediction-error novelty to chance.

**Inference:** combining these, latent prediction error is a principled anomaly score precisely when (i) collapse is controlled and (ii) test data comes from the same feature distribution family as training; both conditions have documented failure cases in the literature above.

---

## 4. Verified prior work applying JEPA-family / latent-predictive models to AD

Every entry below was verified on its own arXiv page (or official repo) on 2026-08-23. Headline numbers are quoted from the papers' own abstracts/pages.

### 4.1 SC-JEPA — time-series anomaly *prediction* (early warning)
- Title/venue: *SC-JEPA: Stabilizing Latent Predictive Learning for Time-Series Anomaly Prediction*, accepted at **SIAM SDM 2026** ([arXiv:2602.04643](https://arxiv.org/abs/2602.04643)).
- Domain: multivariate time series (system-failure early warning).
- Method: JEPA-style latent prediction in a **discretized predictive state space**; a **soft codebook bottleneck** stabilizes latent self-distillation (explicitly motivated by representation-collapse instability of continuous self-distillation on TS); **multi-resolution predictive objective** captures precursors at different temporal scales.
- Benchmarks/results: "five real-world benchmarks … strong and consistent early-warning performance" (abstract gives no numeric table values; not invented here).
- Anomaly scoring: predictive modeling of precursor dynamics in the stabilized latent space (early-warning framing rather than point-score thresholding).

### 4.2 T-SAR-JEPA — temporal anomaly detection in satellite image time series
- Title/venue: *T-SAR-JEPA: Self-Supervised Temporal Anomaly Detection in SAR Amplitude Stacks via Latent Prediction*; won IEEE GRSS Data Fusion Contest 2026; to appear in **IGARSS 2026** ([arXiv:2606.05700](https://arxiv.org/abs/2606.05700); code: [TerraLatent/t-sar-jepa](https://github.com/TerraLatent/t-sar-jepa)).
- Domain: SAR amplitude time stacks (image time series).
- Method: ViT-Base/16 SAR-JEPA encoder domain-adapted on 39,300 Capella patches; a temporal transformer with sinusoidal time encoding **forecasts future latent states from K=7 acquisitions**; progressive unfreezing.
- Benchmark/result: DFC 2026 dataset (300 time-series, 3 AOIs); **ROC-AUC 77.0%** on the Hawaii eruption window vs. RX, PaDiM, Linear AR, LSTM baselines (~50%).
- Scoring: deviation between forecast latent states and encoded actual observations, validated against InSAR coherence as independent pseudo-ground-truth.

### 4.3 Online automotive monitoring with JEPA embeddings
- Title/venue: *Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings*, accepted at **IEEE Intelligent Vehicles Symposium 2026** ([arXiv:2602.09985](https://arxiv.org/abs/2602.09985)).
- Domain: object-state time series from autonomous driving (nuScenes).
- Method: self-supervised **JEBA/JEPA prediction task over object-state sequences** produces embeddings; those embeddings are then fed to "established anomaly detection methods" — i.e., JEPA as representation front-end, classic detector as scorer; designed for label-free operation against unknown anomalies.
- Results: qualitative/quantitative experiments on nuScenes demonstrating framework capability (abstract reports no single headline number).

### 4.4 PULS ("Latent Clarity") — video anomaly anticipation with V-JEPA 2
- Title: *Latent Clarity: Bridging World-Model Kinematics to Semantic Manifolds for Video Anomaly Anticipation* ([arXiv:2607.03558](https://arxiv.org/abs/2607.03558)).
- Domain: continuous video anomaly detection/anticipation.
- Method: distills **V-JEPA 2** physical tensors into a text-aligned semantic hypersphere (KSD Bridge, 490M params) + Anticipatory State Predictor (16.8M params); "Latent Clarity Hypothesis": JEPA's temporal predictor discards aleatoric pixel noise while preserving kinematics, so anticipated future representations are more separable than observed ones.
- Benchmarks/results: **UCF-Crime chunk-level AUROC 0.8994**, **XD-Violence (out-of-distribution) 0.8162** without MIL or hierarchical fusion; Triple-Track Lead-Time protocol with an **L1-surprise gate** yields up to +8.9 pp anticipatory advantage at T−0.5 s (p<0.001).
- Scoring: distance/surprise in the anticipated-latent manifold (L1-surprise gating).

### 4.5 Two further verified, directly-relevant results (smaller scope)
- *Zero-Label Driving Scenario Complexity Detection via Joint Embedding Predictive Architecture* ([arXiv:2606.28383](https://arxiv.org/abs/2606.28383)): minimal JEPA on nuPlan agent-state sequences; **temporal prediction error as zero-shot complexity/anomaly score**; downstream anomaly-detection AP **0.512 vs 0.436 chance**. Evidence both for feasibility and for weakness of raw prediction error alone.
- *Asleep at the Wheel: JEPA's Limitations in Evaluating Novel Driving Data* ([arXiv:2608.01336](https://arxiv.org/abs/2608.01336)): frozen **V-JEPA** encoder + lightweight predictor head; clips flagged by masked-embedding prediction error. Under cross-dataset triage it looks effective, but on a fair single-dataset benchmark it "collapses to chance and is on par with simple no-training baselines"; a lightly supervised probe on the same frozen embeddings almost doubles average precision — the bottleneck is the self-supervised objective, not the representation. **Key negative control for anyone building prediction-error AD scores.**

Adjacent (verified, JEPA-for-TS representation learning but not AD): *CF-JEPA: Mask-free forward prediction … for time-series representation learning* ([arXiv:2606.07031](https://arxiv.org/abs/2606.07031)) argues against disruptive masking for TS continuity; *STST-JEPA* for EEG ([arXiv:2607.06629](https://arxiv.org/abs/2607.06629)) and *CardioState-JEPA* for ECG/PPG/PCG ([arXiv:2608.12944](https://arxiv.org/abs/2608.12944)) are latent-prediction TS foundation-model works; a depth-regularized JEPA world model evaluates "predictor-based surprise detection" in/out-of-domain ([arXiv:2607.16314](https://arxiv.org/abs/2607.16314)).

### 4.6 Classic forecast-then-score lineage (pre-JEPA, verified anchors)
These instantiate the same "predict normality, score deviation" principle in observable space; JEPA moves the prediction into latent space ([LeCun 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf)):
- *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection* ([arXiv:1607.00148](https://arxiv.org/abs/1607.00148)) — prediction/reconstruction error signals for multi-sensor machines.
- *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding* ([arXiv:1802.04431](https://arxiv.org/abs/1802.04431)) — forecast-then-threshold on telemetry (closest classical analogue to SMD-style machine telemetry).
- *MAD-GAN: Multivariate Time Series Anomaly Detection with GANs* ([arXiv:1809.04758](https://arxiv.org/abs/1809.04758)) — generator forecasts, discriminator scores.

---

## 5. Mapping to THIS repo (design inference — nothing below is from the cited literature unless quoted)

Repo facts used (from `AGENTS.md`, `WORKFLOW_PRETEXT.md`): pretext stage trains `ContrastiveModel` (backbone + projection head) with a triplet-style `PretextLoss` over three views (`ts_org` anchor, `ts_w_augment` near — **usually the preceding raw window** for indices>10, `ts_ss_augment` injected sub-anomaly from `SubAnomaly`); clustering metrics (Silhouette↑, Calinski-Harabasz↑, Davies-Bouldin↓) select `model.pth.tar`-family checkpoints via `make_checkpoint`; `contrastive_evaluate` runs MiniBatch K-means (k=2) on embedded views; classification stage (`carla_classification.py`) trains the anomaly head on SMD/PSM.

### 5.1 Option A — latent window-prediction criterion (`pretext_jepa`) *(Inference)*
- Register a **new** criterion name in the `get_criterion` factory (`utils/common_config.py`) instead of reusing `pretext_new`, which the factory rejects (known trap in `AGENTS.md`/`WORKFLOW_PRETEXT.md`).
- Objective (mirroring verified mechanisms): online encoder f_θ embeds a context window; small predictor g_φ predicts the latent of the shifted target window; EMA target encoder f_θ̄ provides detached targets; loss = L2 (I-JEPA-style, [source](https://arxiv.org/html/2301.08243v3)) or L1 (V-JEPA-style, [Eq. 1](https://arxiv.org/html/2404.08471v1)).
- **Data seam already exists:** `AugmentedDataset`'s near view is typically the preceding raw window (`data/custom_dataset.py:59-63`, per `WORKFLOW_PRETEXT.md`), so `(anchor → near)` is naturally a one-step latent forecast; the injected-anomaly view can be kept out of the loss and reserved for probing, or kept as a margin term to preserve the current triplet behavior (hybrid loss).
- Test-time bonus the current pretext lacks: an unsupervised anomaly score `‖f_θ̄(x_{t+k}) − g_φ(f_θ(x_t))‖` computable before any classifier is trained — matching how T-SAR-JEPA and PULS score anomalies (§4.2, §4.4). *(Inference)*

### 5.2 Option B — EMA target encoder added to the existing pipeline *(Inference)*
- Add f_θ̄ updated each step as EMA of f_θ (momentum hyperparameter), targets detached. Implementation surface: optimizer step hook in `utils/train_utils.py`; all tensors created on the configured device — remember the trap that `PretextLoss` defaults tensors to `cuda` (`losses/losses.py:498-530`).
- Collapse safety net: if EMA+stop-grad proves unstable on SMD windows (SC-JEPA documents such instability on TS, [arXiv:2602.04643](https://arxiv.org/abs/2602.04643)), fall back to variance/covariance regularization in the spirit of LeCun's criteria 1–2 ([LeCun 2022 Fig. 13](https://openreview.net/pdf?id=BZ5a1r-kVsf)) or SC-JEPA's soft codebook bottleneck. *(Inference grounded in cited reports)*

### 5.3 Option C — masked-context pretext on SMD windows *(Speculative)*
- Mask a contiguous sub-interval of each window, predict its latent from the visible remainder (1-D translation of I-JEPA's multiblock masking, [source](https://arxiv.org/html/2301.08243v3)); CF-JEPA cautions that masking "disrupt[s] the temporal continuity of time-series signals" ([arXiv:2606.07031](https://arxiv.org/abs/2606.07031)), so compare against mask-free forward prediction (Option A).
- Do **not** implement this inside `PretextLoss.random_crop` — that method does not exist and `crop: True` crashes on the first batch (confirmed bug, `losses/losses.py:573-581`). A separate mask collator keeps the known trap dormant.

### 5.4 Checkpointing and evaluation interplay *(Inference)*
- The clustering-metric machinery can stay untouched: it operates on whatever embeddings `contrastive_evaluate` produces, so JEPA-trained embeddings remain compatible with Silhouette/CH/DB selection.
- However, clustering metrics measure view-clusterability, **not predictive quality**; a JEPA run should additionally track validation latent-prediction loss as a selection signal. That means touching `make_checkpoint` best-value sentinels, metric directions, and the resume metadata (`next_epoch`, scheduler serialization) that `WORKFLOW_PRETEXT.md` records as recently-fixed — extend the new checkpoint format; do not regress legacy-resume handling (open TODO item there).
- Serendipity: validation datasets are plain SMD windows supplying only `ts_org` (`carla_pretext.py:62-64`) — the exact input a pure prediction loss needs. A JEPA validation loss therefore *fits* the existing asymmetric validation set instead of fighting it (contrast the documented three-view vs one-view evaluation mismatch). *(Inference)*

### 5.5 Risks mapped to known traps (`AGENTS.md` / `WORKFLOW_PRETEXT.md`)
| Trap | Impact on a JEPA port |
|---|---|
| `crop: True` crash (no `random_crop`) | Any window-masking must bypass `PretextLoss`; keep `crop: False` in configs |
| Criterion factory whitelist | New name required in `get_criterion`; avoid unsupported `criterion_kwargs` pattern |
| Device default `cuda` in loss state | Create EMA/predictor buffers on `cfg.device`; CPU configs break silently otherwise |
| Backbone `kernel_sizes` vs `mid_channels` mismatch | Unchanged requirement; JEPA adds a predictor head, not new backbone blocks |
| Train/eval mode mismatch for negative branch | Apply uniform `.train()/.eval()` policy across context/target branches (BatchNorm/dropout) |
| View-semantics uncertainty (near = preceding raw window) | This ambiguity *becomes the feature* under Option A, but quantify it first (open TODO in `WORKFLOW_PRETEXT.md`) |
| O(N²) silhouette/memory at evaluation | Unchanged; adding prediction-loss logging is cheap relative to it |
| Hardcoded seed / no worker seeds | Same reproducibility caveats apply to any new stochastic masking |

### 5.6 Evidence-based cautions for the experiment design *(Inference from §4)*
- Raw prediction-error scores can be weak or confounded: AP 0.512 vs 0.436 chance in [one verified study](https://arxiv.org/abs/2606.28383), and chance-level under a fair protocol in [another](https://arxiv.org/abs/2608.01336). Pair the latent prediction error with embedding-based scores (this repo's k-means/silhouette tooling) rather than replacing them wholesale.
- Stabilization is the crux on time series (collapse/instability reported by [SC-JEPA](https://arxiv.org/abs/2602.04643)); plan an ablation axis: {triplet baseline} × {+EMA latent prediction} × {+codebook or VICReg-style regularizer}.
- Hierarchical H-JEPA-style multi-scale stacking (short-horizon JEPA feeding a longer-horizon one; [LeCun 2022 §4.6](https://openreview.net/pdf?id=BZ5a1r-kVsf)) is a natural later step for SMD's long recordings, and matches SC-JEPA's multi-resolution motivation. Action-conditioned variants (V-JEPA 2-AC) do not map — SMD/PSM have no action channel. *(Speculative)*

---

## 6. Limitations and open questions

- **OpenReview access:** openreview.net served a CAPTCHA wall on 2026-08-23 (forum, pdf, and API endpoints); LeCun 2022 content was read from a Wayback snapshot of the official PDF. Canonical citations point to OpenReview.
- **Meta blogs:** ai.meta.com returns HTTP 400 to scripted clients; official blog content was read via Wayback snapshots of the canonical URLs.
- **Negative search result:** no primary-source JEPA paper for **industrial image defect detection** (e.g., MVTec) was found — `abs:"JEPA" AND abs:"anomaly detection"` (8 hits, none industrial-image), `all:"JEPA" AND all:"MVTec"` (0 hits), `all:"defect detection" AND abs:"JEPA"` (0 hits) on the arXiv API, 2026-08-23. Absence of evidence ≠ evidence of absence, but nothing is claimed in that direction here.
- **Unverified / excluded:** (i) a standalone Meta AI blog post specifically for V-JEPA (2024) — canonical URL not verified in this session, so not cited; (ii) headline benchmark tables of SC-JEPA (numbers beyond the abstract) — abstract-only verification; (iii) venue status of several 2026 arXiv preprints listed as "accepted at" only via their own arXiv comments field.
- **Open question (methodological):** does latent-prediction error add signal *on top of* the injected-anomaly contrastive signal this repo already uses? None of the verified papers combines both; that experiment would be novel here. *(Inference)*
- **Open question (theory):** LeCun specifies energy-minimizing inference over latent z for multi-modal futures ([LeCun 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf)); practical TSAD works use deterministic predictors with no z, so the score ignores multi-modality. Whether a latent-variable predictor improves TSAD scores is untested in the verified literature.

## 7. Sources (all accessed 2026-08-23)

| # | Source | Type | URL |
|---|---|---|---|
| 1 | LeCun, *A Path Towards Autonomous Machine Intelligence* (2022) | Paper (OpenReview) | https://openreview.net/forum?id=BZ5a1r-kVsf (canonical; blocked by bot-check; content read via https://web.archive.org/web/2023id_/https://openreview.net/pdf?id=BZ5a1r-kVsf ) |
| 2 | Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA), CVPR 2023 | Paper | https://arxiv.org/abs/2301.08243 ; full text https://arxiv.org/html/2301.08243v3 |
| 3 | facebookresearch/ijepa (official repo, archived 2024-08-01) | Repo | https://github.com/facebookresearch/ijepa |
| 4 | Meta AI blog, *I-JEPA: The first AI model based on Yann LeCun's vision* (2023) | Blog | https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/ (via https://web.archive.org/web/20230712232736/https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/ ) |
| 5 | Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA) | Paper | https://arxiv.org/abs/2404.08471 ; full text https://arxiv.org/html/2404.08471v1 |
| 6 | Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* | Paper | https://arxiv.org/abs/2506.09985 |
| 7 | facebookresearch/vjepa2 (official repo incl. V-JEPA 2-AC, energy-landscape notebook, V-JEPA 2.1 notes) | Repo | https://github.com/facebookresearch/vjepa2 ; notebook: https://github.com/facebookresearch/vjepa2/blob/main/notebooks/energy_landscape_example.ipynb |
| 8 | Meta AI blog, *V-JEPA 2 world model benchmarks* (2025) | Blog | https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks (via https://web.archive.org/web/202506120249/https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/ ) |
| 9 | He, Wen, Wang, Ma, *SC-JEPA* (SDM 2026) | Paper | https://arxiv.org/abs/2602.04643 |
| 10 | Woldesenbet & Woldesenbet, *T-SAR-JEPA* (IGARSS 2026) | Paper | https://arxiv.org/abs/2606.05700 ; repo https://github.com/TerraLatent/t-sar-jepa |
| 11 | *Online Monitoring Framework for Automotive Time Series Data using JEPA Embeddings* (IEEE IV 2026) | Paper | https://arxiv.org/abs/2602.09985 |
| 12 | *Latent Clarity: … Video Anomaly Anticipation* (PULS) | Paper | https://arxiv.org/abs/2607.03558 |
| 13 | *Asleep at the Wheel: JEPA's Limitations in Evaluating Novel Driving Data* | Paper | https://arxiv.org/abs/2608.01336 |
| 14 | *Zero-Label Driving Scenario Complexity Detection via JEPA* | Paper | https://arxiv.org/abs/2606.28383 |
| 15 | *CF-JEPA: Mask-free forward prediction … time-series representation learning* | Paper | https://arxiv.org/abs/2606.07031 |
| 16 | Malhotra et al., *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection* | Paper | https://arxiv.org/abs/1607.00148 |
| 17 | Hundman et al., *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding* | Paper | https://arxiv.org/abs/1802.04431 |
| 18 | Li et al., *MAD-GAN: Anomaly Detection with GANs for Multivariate Time Series* | Paper | https://arxiv.org/abs/1809.04758 |
| 19 | Local grounding: `AGENTS.md`, `WORKFLOW_PRETEXT.md` (this repo) | Doc | repo root |
