# RESEARCH: Fourier Transforms in Deep Time-Series Models — Time-Domain vs Frequency-Domain Quality, Hybrid Architectures, and Anomaly Detection

Date: 2026-08-31. Scope: research only. Every factual claim carries a primary-source citation (paper page/PDF + official code where available); anything not traceable to a primary source is labeled **Inference:** or **Speculative:** or excluded (see §7). All cited URLs were retrieved during this session — most by direct fetch, a few (flagged in §7/Sources) via the session search tool's snapshot of the primary page.
Grounding for §6: local `AGENTS.md`, grep of this repo for `fft` usage, and the sibling note `RESEARCH_TS_AD_HEADS.md`.

---

## 1. TL;DR

- The DFT/FFT enters deep time-series models in **at least six distinct roles**: (1) unparameterized global *token mixing* in transformers (FNet); (2) *periodicity discovery* to reshape 1D series into 2D tensors for CNN-style modeling (TimesNet); (3) *sparse frequency selection* replacing attention for O(L) complexity (FEDformer; Autoformer's FFT-computed autocorrelation; FourierGNN's Fourier graph operator); (4) *pure frequency-domain learning* on complex coefficients (FreTS; FITS); (5) *noise removal / energy compaction* (FiLM's Fourier Enhanced Layer; FITS's low-pass filter); (6) *frequency views, features or augmentations* feeding time-domain objectives (SR-CNN spectral residual, CoST seasonal contrastive loss, TF-C frequency encoder, FCVAE frequency condition, TFAD/FreCT/ATF-UAD/TimeVQVAE-AD frequency branches).
- **There is no clean "time-domain beats frequency-domain" (or vice versa) quality answer, and the SOTA literature mostly does not force the choice**: the best-reported results come from *hybrid* designs that use frequency for what it is good at (global/periodic structure, noise immunity, compactness) and stay in the time domain for supervision, localization, and event-like content. Even "frequency-domain" models such as FITS and FreTS invert back to the time domain for output/supervision — FITS states it "fundamentally remains a time domain model" [S6].
- **Verified head-to-head AD numbers** (both from primary sources, both on SMD/MSL/SMAP/SWaT/PSM):
  - TimesNet (FFT→2D, ICLR 2023, Table 5): avg F1 **86.34**, vs FEDformer (frequency-enhanced) 84.97, Autoformer (FFT autocorrelation) 84.26, DLinear 82.46, Anomaly Transformer (time-only) 80.50, vanilla Transformer **76.88** [S3].
  - FITS (ICLR 2024, Table 6): SMD F1 **99.95** vs TimesNet 85.81 and Anomaly Transformer 92.33; SWaT 98.9 vs 91.74/94.07; but SMAP **70.74** vs Anomaly Transformer 96.69 and MSL 78.12 vs 93.59 — FITS's own analysis: frequency-domain representation is *worse* on binary/event-style data, where "time-domain modeling is preferable" [S6].
- **Both-modality architectures exist in every family** (forecasting, representation learning, AD): FEDformer, Autoformer, TimesNet, FITS, FreTS, FiLM, CoST, TF-C, TFAD, FCVAE, FreCT, ATF-UAD, TimeVQVAE-AD — see §4 for the mechanism each uses.
- **AD-specific**: frequency-informed and hybrid models dominate on *periodic monitoring* data (SMD, SWaT, PSM); pure time-domain models remain competitive or better on event/impulse data (SMAP, MSL); a 2026 five-family benchmark concludes "no single inductive bias dominates" [S15]. Threshold/evaluation leakage is pervasive in published AD numbers (test-set thresholds; see §5.4) — treat cross-paper quality claims cautiously; this repo's clean-train-only thresholding is the correct hygiene.
- **For THIS repo** (JEPA latent-prediction TSAD): FFT currently appears only in the metrics layer (`acf(fft=True)`, `np.fft.fft` in VUS code), never in models or losses [§6]. Cheap, verified-in-literature additions to consider (all design inference): TimesNet-style FFT period discovery to drive masking, TF-C-style time/frequency consistency as an auxiliary pretext loss, FITS-style low-pass-filtered targets, or an SR-style spectral-residual scoring signal.

---

## 2. The six roles of the Fourier transform in deep TS models

The DFT is a linear, invertible, O(N log N) map from a length-N real series to N/2+1 complex Fourier coefficients (amplitude + phase). PyTorch ships it first-party as `torch.fft` (`fft`, `rfft`, `irfft`, `fftfreq`, `fftshift`, …) [S1] — no exotic dependency is needed to use any of the mechanisms below. Verified roles in the literature:

| # | Role | Mechanism | Exemplar (verified) |
|---|---|---|---|
| 1 | Token mixing | Unparameterized 2D DFT (sequence × hidden) replaces self-attention; real part kept | FNet [S2] |
| 2 | Periodicity discovery | FFT → top-k amplitudes → period lengths → reshape 1D series into k 2D tensors (rows=inter-period, cols=intra-period) → 2D kernels | TimesNet [S3] |
| 3 | Sparse/global attention replacement | DFT → keep M randomly selected modes (incl. high-frequency) → learnable complex weights → zero-pad → IDFT; O(L) | FEDformer FEB-f/FEA-f [S4]; Autoformer autocorrelation via FFT [S5]; FourierGNN Fourier Graph Operator [S10] |
| 4 | Learning on complex coefficients | rFFT → complex-valued linear layer (learns amplitude scaling + phase shift) → zero-pad → irFFT; or DFT → real/imag MLPs | FITS [S6]; FreTS [S9] |
| 5 | Noise removal / energy compaction | Keep low-frequency Fourier components + low-rank projection; or low-pass filter before learning | FiLM FEL [S7]; FITS LPF [S6] |
| 6 | Frequency views/features/augmentations | Frequency branch, spectral features, or frequency-space augmentations feeding a time-domain objective | SR-CNN [S8], TF-C [S11], CoST [S12], FCVAE [S13], TFAD [S14], FreCT [S16], ATF-UAD [S17], TimeVQVAE-AD [S18] |

Details of the canonical exemplars:

- **FNet** (NAACL 2022): replaces the self-attention sublayer with a 2D DFT (one 1D DFT along the sequence dimension, one along the hidden dimension), keeping only the real part. Claims 92–97% of BERT's GLUE accuracy while training 80% faster on GPUs / 70% on TPUs; works without position embeddings. The paper's own interpretation: "multiplying by the feed-forward sublayer coefficients in the frequency domain is equivalent to convolving … in the time domain", i.e., FNet alternates between time and frequency views per block [S2].
- **TimesNet** (ICLR 2023): "we analyze the time series in the frequency domain by Fast Fourier Transformer (FFT)"; selects the top-k amplitudes as the most significant frequencies, giving k period lengths; pads and reshapes the 1D series into k 2D tensors so intra-period variation runs along columns and inter-period variation along rows; a parameter-efficient inception block (2D CNN) models them; results are fused and added back as a residual. The motivation: "the original 1D structure of time series can only present the variations among adjacent time points" [S3].
- **FEDformer** (ICML 2022): replaces self-attention (encoder) and cross-attention (decoder) with Frequency Enhanced Blocks/Attention (FEB-f/FEA-f): DFT → keep only M randomly selected modes (random, not just low-frequency — "a randomly selected subset of frequency components, including both low and high ones, will give a better representation", with theoretical justification) → elementwise multiply by learnable complex parameters → zero-pad → IDFT. This gives "linear computational complexity and memory cost" vs quadratic attention. Wavelet variants (FEB-w/FEA-w) also provided [S4].
- **Autoformer** (NeurIPS 2021): keeps a time-domain transformer skeleton but replaces self-attention with Auto-Correlation, computed via FFT ("We utilize the Fast Fourier Transform to calculate the autocorrelation R(τ)"); period lengths are chosen from autocorrelation peaks and similar sub-series are aggregated by time-delay rolling. O(L log L). Its series-decomposition blocks (moving average) separate trend from seasonal inside the network [S5].
- **FITS** (ICLR 2024 Spotlight): rFFT → reversible instance norm (RIN) → low-pass filter (LPF) → a single complex-valued linear layer ("learn amplitude scaling and phase shifting as the multiplication of complex numbers") → zero-pad → irFFT; supervised in the time domain with MSE. ~10k parameters (1–4k in the AD setting); "50 times smaller than the lightweight temporal linear model DLinear and approximately 10,000 times smaller than other mainstream models". The paper is explicit that it "fundamentally remains a time domain model" — the frequency domain is a compute/parameter-efficient internal representation, not the output space [S6].
- **FreTS** (NeurIPS 2023): domain conversion via DFT; redesigned complex MLPs (separate real and imaginary mappings) run on both inter-series (channel) and intra-series (temporal) scales; final predictions recovered from frequency components. Claims two learned advantages: "(i) global view … (ii) energy compaction" [S9].
- **FiLM** (NeurIPS 2022): Legendre polynomial projections (from the LMU line) preserve history; a Fourier Enhanced Layer keeps "the part of the representation related to low-frequency Fourier components and the top eigenspace to remove the impact of noises". Ablations show FEL beats MLP/LSTM/CNN/attention variants of the same module [S7].
- **FourierGNN** (NeurIPS 2023): represents a multivariate window as a "hypervariate graph" (every value = a node) and performs recursive matrix multiplications in Fourier space via the Fourier Graph Operator; a proven equivalence: the FGO "is equivalent to graph convolutions in the time domain". Log-linear complexity [S10].

---

## 3. Time-domain vs frequency-domain: what the primary sources actually claim

### 3.1 Claimed advantages of frequency-domain processing (verified wording)

- **Global view / global dependencies**: FEDformer — time-domain point-wise attention "fails to maintain the global property and statistics of time series as a whole" and frequency-domain attention is "much sparser … [and] can represent the signal more compactly" [S4]. FreTS — "frequency spectrum makes MLPs own a complete view for signals and learn global dependencies more easily" [S9].
- **Energy compaction / noise immunity**: FreTS — "frequency-domain MLPs concentrate on smaller key part of frequency components with compact signal energy" [S9]. FiLM — Fourier projection is used explicitly "to remove noise" / "minimize the impact of noisy signals" [S7]. FITS — the LPF discards high-frequency components which "typically comprise noise"; a filtered waveform "exhibits minimal distortion even when only preserving a quarter of the original frequency domain representation" [S6].
- **Sparsity → efficiency**: FEDformer's random-mode selection → linear complexity [S4]; Autoformer O(L log L) [S5]; FourierGNN log-linear [S10]; FNet 80%/70% training speedups [S2].
- **Periodicity is load-bearing for AD**: TimesNet's AD discussion — "by taking the periodicity into consideration, TimesNet, FEDformer and Autoformer all achieve great performance", while "the vanilla attention mechanism calculates the similarity between each pair of time points, which can be distracted by the dominant normal time points" (vanilla Transformer avg F1 76.88 vs TimesNet 86.34 on the same reconstruction-criterion protocol) [S3].

### 3.2 Claimed advantages of time-domain processing (verified wording)

- **Event/binary/impulse data**: FITS's own analysis of its AD table — "FITS shows comparatively lower performance on the SMAP and MSL datasets. These datasets present a challenge due to their binary event data nature, which may not be effectively captured by FITS' frequency domain representation. In such cases, time-domain modeling is preferable as the raw data format is sufficiently compact" [S6].
- **Local/point-wise structure**: TimesNet motivates its 2D design by the converse: "the original 1D structure of time series can only present the variations among adjacent time points" — i.e., pure frequency representations lose adjacent-point locality unless combined with time-space structure [S3].
- **Supervision is naturally in time**: FITS is supervised in the time domain (MSE after irFFT) even though it operates in frequency internally; "This end-to-end design enables FITS to adapt to various downstream tasks with commonly-used time domain supervision" [S6].

### 3.3 Verified head-to-head numbers

**Anomaly detection (F1 %, reconstruction-error criterion, SMD/MSL/SMAP/SWaT/PSM)** — TimesNet ICLR 2023 Table 5 (via the paper's ar5iv HTML; see §7 for snapshot caveat) [S3]:

| Model | SMD | MSL | SMAP | SWaT | PSM | Avg F1 |
|---|---|---|---|---|---|---|
| TimesNet (ResNeXt backbone) | 85.81 | 85.15 | 71.52 | 91.74 | 97.47 | **86.34** |
| FEDformer (frequency-enhanced) | 85.08 | 78.57 | 70.76 | 93.19 | 97.23 | 84.97 |
| Autoformer (FFT autocorrelation) | 85.11 | 79.05 | 71.12 | 92.74 | 93.29 | 84.26 |
| DLinear (time-domain linear) | 77.10 | 84.88 | 69.26 | 87.52 | 93.55 | 82.46 |
| Anomaly Transformer (time-domain) | 85.49 | 83.31 | 71.18 | 83.10 | 79.40 | 80.50 |
| Vanilla Transformer (time-domain) | 79.56 | 78.68 | 69.70 | 80.37 | 76.07 | **76.88** |

Reading (per the paper's own text): all three *frequency-informed* models (TimesNet, FEDformer, Autoformer) outperform the pure time-domain attention models on average; the gap to vanilla Transformer is ~10 F1 points. Note these numbers share one criterion (reconstruction error) and one protocol — a comparatively clean comparison.

**Anomaly detection (F1 %)** — FITS ICLR 2024 Table 6 (camera-ready; FITS repo later re-ran after a bug fix and flagged leakage in some baselines — see §5.4) [S6]:

| Model | SMD | PSM | SWaT | SMAP | MSL |
|---|---|---|---|---|---|
| FITS (frequency interpolation) | **99.95** | 93.96 | **98.9** | 70.74 | 78.12 |
| TimesNet | 85.81 | 97.47 | 91.74 | 71.52 | 85.15 |
| Anomaly Transformer (time-only) | 92.33 | 97.89 | 94.07 | **96.69** | **93.59** |
| THOC | 84.99 | **98.54** | 85.13 | 90.68 | 89.69 |
| OmniAnomaly | 85.22 | 80.83 | 82.83 | 86.92 | 87.67 |
| LightTS / DLinear (time-domain) | 82.53 / 77.1 | 97.15 / 93.55 | 93.33 / 87.52 | 69.21 / 69.26 | 78.95 / 84.88 |

Reading (per FITS's own text): a nearly-pure frequency model wins big on the strongly periodic datasets (SMD, SWaT) but loses on event-style data (SMAP, MSL) where the time-domain Anomaly Transformer is best. **Neither domain wins outright; the winning choice is dataset-dependent.**

**NLP transfer (for calibration of "how much quality does the domain cost")**: FNet — 92–97% of BERT accuracy on GLUE for ~80%/70% training speedup; i.e., dropping attention entirely for a parameter-free DFT costs a few accuracy points, not a collapse [S2].

**Representation-learning quality**: CoST (time+frequency contrastive) beats the best end-to-end forecasting approach by 39.3%/18.22% MSE (multi/univariate) and the next feature-based approach by 21.3%/4.71% [S12]. TF-C (time+frequency consistency pre-training) outperforms 8 SOTA baselines by 15.4% avg F1 in one-to-one transfer, 8.4% precision in one-to-many [S11].

### 3.4 Benchmark-level (cross-family) evidence

A 2026 preprint benchmarking ten detectors spanning statistical / reconstruction / association / frequency / generic-transformer families on SMD, MSL, SMAP, PSM, MSDS concludes: "no single inductive bias dominates"; per-dataset winners vary; "Strongly periodic data still favours frequency-aware processing" (TimesNet wins PSM), while channel-graph/association-style detectors win elsewhere; its spectral-branch ablation costs up to −19.3 pt (VUS-ROC) on SMAP [S15]. **Inference:** consistent with §3.3 — frequency helps where periodicity is informative and hurts where it is not.

---

## 4. Architectures that use BOTH time and frequency modalities (verified)

| Model (venue) | Time-domain part | Frequency-domain part | Output/supervision | Source |
|---|---|---|---|---|
| FEDformer (ICML 2022) | Transformer encoder–decoder, decomposition blocks | FEB-f/FEA-f replace attention: DFT → random M modes → complex weights → IDFT (or wavelet) | Time domain | [S4] |
| Autoformer (NeurIPS 2021) | Transformer skeleton + series decomposition blocks | FFT computes autocorrelation; period-based time-delay aggregation replaces attention | Time domain | [S5] |
| TimesNet (ICLR 2023) | 1D residual stream | FFT → top-k periods → 1D→2D reshape; 2D inception CNN models intra/inter-period variation | Time domain (reconstruction) | [S3] |
| FITS (ICLR 2024) | RIN normalization; time-domain MSE supervision | rFFT → LPF → complex linear layer (amplitude/phase) → zero-pad → irFFT | Time domain | [S6] |
| FreTS (NeurIPS 2023) | Input projection; final inversion | DFT → frequency MLPs on real/imag parts, channel + temporal scales | Time domain (after inversion) | [S9] |
| FiLM (NeurIPS 2022) | Legendre projection units (history memory) | Fourier Enhanced Layer: low-frequency Fourier + low-rank noise removal | Time domain | [S7] |
| CoST (ICLR 2022) | Trend Feature Disentangler + time-domain contrastive loss | Seasonal Feature Disentangler: FFT → per-frequency complex linear layer → iFFT; amplitude+phase contrastive losses | Time-domain representations | [S12] |
| TF-C (NeurIPS 2022) | Contrastive time encoder (time augmentations) | Contrastive frequency encoder (frequency-spectrum augmentations) + cross-space projectors; consistency loss L = λ(L_T+L_F)+(1−λ)L_C | Concatenated time+freq embedding | [S11] |
| TFAD (CIKM 2022) | Time branch (TCN) | Frequency branch (TCN on interleaved Re/Im DFT coefficients); + decomposition, augmentation | Time-domain scores | [S14] |
| FCVAE (WWW 2024) | CVAE encoder–decoder reconstructing in time | Global + local frequency features (GFM/LFM) as the CVAE *condition*; target attention over frequency | Time-domain reconstruction error | [S13] |
| FreCT (2025 preprint) | Patch → transformer + conv encoder (KL-divergence consistency in time) | Fourier transform of embeddings; modulus-based deviation in frequency | Time+freq consistency scores | [S16] |
| ATF-UAD (Neural Networks 2023) | Time reconstructor (parity sampling + attention/GCN) | Frequency reconstructor (FT, reconstructs anomalous frequency bands); dual-view adversarial learning | Combined residuals | [S17] |
| TimeVQVAE-AD (2023 preprint) | Masked generative prior over quantized tokens | Tokenization via STFT: tokens live on a time×frequency grid; NLL-based scores per frequency band | NLL anomaly scores (band-factorizable) | [S18] |
| SR-CNN (KDD 2019) | CNN discriminator on SR output | Spectral Residual: FFT → log-amplitude minus smoothed average → iFFT saliency map | Saliency-based scores | [S8] |
| FourierGNN (NeurIPS 2023) | Hypervariate graph over window values | DFT → recursive Fourier Graph Operator multiplications → IDFT (provably ≡ graph convolution in time) | Time domain | [S10] |

**Pattern (verified from the papers themselves):** the winning designs do *not* choose a single domain — they route *structure discovery* (periodicity, global dependencies, noise separation) through the frequency domain and keep *supervision, localization, and output* in the time domain. FITS's self-description ("fundamentally remains a time domain model") and FreTS's "Domain Conversion/Inversion" pipeline are the clearest statements of this.

---

## 5. Anomaly detection specifically

### 5.1 Frequency-front-end AD

- **SR-CNN (Microsoft, KDD 2019)**: the first transfer of Spectral Residual from visual saliency to time-series AD. SR = FFT → log amplitude spectrum → subtract its smoothed average → inverse FFT; anomalies are the salient residuals. A CNN then replaces the single threshold, trained on *synthetic* anomalies injected into SR output. Verified results: on the KPI dataset, adding the SR feature lifts a supervised DNN from F1 0.798 → 0.811 (+1.6%); production service used by "more than 200 teams within Microsoft … 4 million time-series per minute" [S8]. (The Azure Anomaly Detector shipped this algorithm — Microsoft blog + ML.NET `SrCnnAnomalyEstimator` [S8].)
- **FITS AD**: reconstruction task = recover the original segment from a downsampled one via frequency interpolation; scores from reconstruction error in time. Results in §3.3 [S6].

### 5.2 Hybrid time+frequency AD (both modalities, all verified)

- **TFAD (CIKM 2022)**: parallel time branch (TCN) and frequency branch (TCN over interleaved real/imaginary DFT coefficients), plus decomposition and augmentation; claims SOTA on univariate and multivariate benchmarks [S14].
- **FCVAE (WWW 2024)**: frequency (global + local) conditions a CVAE's reconstruction; reports beating Anomaly Transformer and other SOTA on Yahoo/KPI/NAB/WSD, plus a production deployment at a cloud system (F1/F1* improvements of ~10.9–11.1% over its legacy detector) [S13].
- **FreCT (2025)**: transformer+convolution encoder for time-domain consistency (KL divergence, stop-gradient) + Fourier modulus deviation in frequency; explicitly motivated by MSE loss being "largely magnified by anomaly segments" [S16].
- **ATF-UAD (Neural Networks 2023)**: time reconstructor + frequency reconstructor (reconstructs anomalous frequency bands) + dual-view adversarial learning; avg +6.94% F1 over SOTA on 9 datasets [S17].
- **TimeVQVAE-AD (2023)**: masked generative modeling over STFT-derived tokens (time×frequency grid); a learned prior scores subsequence normality by negative log-likelihood; frequency-band anomaly factorization for explainability — the closest published relative to a *masked-prediction (JEPA-style) objective that operates on a time–frequency tokenization* [S18].

### 5.3 Time-domain SOTA baselines (for the comparison)

- **Anomaly Transformer (ICLR 2022)**: association-discrepancy head (prior-association vs series-association) — entirely time-domain attention; still the best on SMAP/MSL in FITS's Table 6 [S6]; local note `RESEARCH_TS_AD_HEADS.md` §7.1 covers its mechanism and its test-statistic thresholding.
- **TranAD (VLDB 2022)**: transformer encoder–decoder with self-conditioning (phase-2 input = squared error of phase-1) and adversarial training; claims up to +17% F1 and −99% training time vs baselines [S19]. **Verified in its official code** (`main.py`, both `main` and `master` branches fetched this session): the loss is pure MSE over the two phases — **no Fourier/frequency term exists in the official implementation**. TranAD is a time-domain model; do not cite it as frequency-based. (A 2023 paper re-evaluating it on UCR-TSA reports collapse to ~0.16 avg accuracy — see §5.4.)

### 5.4 Evaluation-integrity caveats (primary sources, relevant to any quality comparison)

- FITS's official repo "Important Notice": "In previous anomaly detection works, anomaly threshold is calculated based on the test_set … such setting may violate the assumption that the test_set should be unavailable before deploying the model. Such method may cause information leakage and cherrypicked result on the test_set" — pointing at Anomaly Transformer's thresholding pipeline; FITS instead picks thresholds on the validation set. The same repo flags a bug that affected a wide range of results and documents the re-runs [S6].
- **The Elephant in the Room (NeurIPS 2024 D&B)**: under a standardized protocol, published SOTA AD methods (TranAD among them) show dramatically worse performance — TranAD's NAB/NASA F1 claims (0.94/0.89) collapse to ~0.16 average accuracy on UCR-TSA under re-evaluation [S20]. **Inference:** cross-paper AD "quality" comparisons are unreliable without protocol re-runs; this repo's clean-train-only thresholds + frozen metrics stack are the right posture.

---

## 6. Relevance to this repo (local grounding + design inference)

**Local grounding (verified this session by grep):** FFT appears in this repo only in the metrics layer, never in models or losses:
- `metrics/evaluate_utils.py:30` and `metrics/vus/utils/slidingWindows.py:14` — `acf(data, nlags=400, fft=True)` (statsmodels autocorrelation for VUS metrics).
- `metrics/vus/models/distance.py:476-477` — `np.fft.fft` in the (non-parametric) VUS distance computations.
- No `torch.fft`/`rfft`/`irfft` anywhere in `models/`, `losses/`, or the stage scripts. The dense-L1 latent-prediction JEPA pipeline is entirely time-domain.

**Design inference (labeled; not claims about the literature):** given the verified literature above, the following would be low-risk, high-signal ways to bring frequency information into this repo's pretext/adapt/score pipeline, each with a primary-source template to copy the mechanism from:
1. **Period-aware masking (TimesNet-style)**: run `torch.fft.rfft` on pretrain windows, pick top-k periods, and bias masking toward aligned phase positions so the predictor must learn inter-period consistency — directly parallels TimesNet's "discover the multi-periodicity adaptively" [S3] and fits the existing masked-corpus stage.
2. **Time–frequency consistency auxiliary loss (TF-C-style)**: add a small frequency-view encoder (rFFT of the input window or of latents → MLP) and a consistency term between time-domain and frequency-domain latent views during pretraining [S11] — mirrors TF-C's L_C term and CoST's seasonal contrastive loss [S12].
3. **Low-pass-filtered targets (FITS-style)**: reconstruct/predict from a filtered (LPF) input, forcing the encoder to be robust to the high-frequency noise that FITS argues is "typically comprise[d] [of] noise" [S6]; or a complex-linear predictor head as in FITS/FreTS [S6][S9].
4. **Frequency-domain scoring signal (SR-style)**: spectral-residual saliency (log-amplitude minus smoothed mean, via `torch.fft`) as an auxiliary per-position score fused at the Scorer level — the mechanism behind Microsoft's production detector [S8].
5. **Caution from the literature**: FITS's own failure mode on binary/event-style channels (SMAP/MSL) [S6] and the CCG-MSD benchmark's "no single bias dominates" [S15] argue for keeping the time-domain path primary and treating any frequency component as an auxiliary signal, never a replacement — consistent with every hybrid design in §4.

---

## 7. Limitations, dead ends, unverifiable

- **TranAD "Fourier loss" is NOT in the official code.** Both `main.py` (branches `main` and `master`, fetched directly) implement the loss as pure MSE over the two decoder phases; no `fft`/frequency term exists. Any secondary source claiming a Fourier loss for TranAD should be treated as unverified — this note makes no such claim.
- **TimesNet Table 5 numbers** were read from the ar5iv HTML rendering of the ICLR paper (session search snapshot), not the PDF directly; treat individual cells as high-confidence but single-snapshot. The paper's abstract-level SOTA claims were verified from the official repo README and ICLR page.
- **FITS Table 6 numbers** come from the ICLR 2024 camera-ready PDF (fetched via proceedings.iclr.cc) and the paper's arXiv v3 HTML; the official repo documents a post-publication bug fix and re-runs, and flags threshold leakage in compared baselines (§5.4) — the SMD 99.95 / SMAP 70.74 figures should be read with that context.
- **Accessed via session-search snapshots only (primary pages, not direct fetch):** FreCT (arXiv HTML snippet), ATF-UAD (ACM DL snippet), TimeVQVAE-AD (arXiv PDF snippet), Elephant in the Room (NeurIPS PDF snippet), CCG-MSD (arXiv HTML snippet, 2026 preprint). Treat quoted wording from these as high-confidence but single-snapshot; CCG-MSD additionally is a preprint, not peer-reviewed.
- **CoST venue**: official repo says ICLR 2022; arXiv record is 2202.01575 — cited as ICLR 2022 per the official repo's bibtex.
- **Not verified / excluded from claims:** any per-dataset AD F1 for TFAD/FCVAE beyond abstract-level claims; "pure frequency-domain" end-to-end models — none found; the closest (FITS) explicitly self-describes as a time-domain model [S6]. FEDformer's wavelet variant details verified at mechanism level only.
- **Wrong-ID trap caught during research:** arXiv 2205.08897 (not 2202.09329) is FiLM's correct ID; 2202.09329 is a different paper.

---

## 8. Sources (all accessed 2026-08-31)

| # | Source | Type | URL |
|---|---|---|---|
| 1 | PyTorch `torch.fft` module docs (v2.13) | First-party API docs | https://docs.pytorch.org/docs/2.13/fft.html |
| 2 | Lee-Thorp et al., FNet, NAACL 2022 | Paper (arXiv + ACL Anthology) | https://arxiv.org/abs/2105.03824 ; https://aclanthology.org/2022.naacl-main.319.pdf ; code https://github.com/google-research/google-research/tree/master/f_net |
| 3 | Wu et al., TimesNet, ICLR 2023 | Paper (arXiv + ICLR) + official repo | https://arxiv.org/abs/2210.02186 ; https://ar5iv.labs.arxiv.org/html/2210.02186 ; https://openreview.net/pdf?id=ju_Uqw384Oq ; https://github.com/thuml/TimesNet |
| 4 | Zhou et al., FEDformer, ICML 2022 | Paper (arXiv + PMLR PDF) + official repo | https://arxiv.org/abs/2201.12740 ; https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf ; https://github.com/DAMO-DI-ML/ICML2022-FEDformer |
| 5 | Wu et al., Autoformer, NeurIPS 2021 | Paper (arXiv + NeurIPS PDF) + official repo | https://arxiv.org/abs/2106.13008 ; https://proceedings.neurips.cc/paper/2021/file/bcc0d400288793e8bdcd7c19a8ac0c2b-Paper.pdf ; https://github.com/thuml/Autoformer |
| 6 | Xu, Zeng, Xu, FITS, ICLR 2024 Spotlight | Paper (arXiv v3 + ICLR proceedings PDF) + official repo | https://arxiv.org/abs/2307.03756 ; https://proceedings.iclr.cc/paper_files/paper/2024/file/701251e1db4a2e4dd2ef23f5265d5936-Paper-Conference.pdf ; https://github.com/VEWOXIC/FITS |
| 7 | Zhou et al., FiLM, NeurIPS 2022 | Paper (arXiv + NeurIPS PDF) + official repo | https://arxiv.org/abs/2205.08897 ; https://proceedings.neurips.cc/paper_files/paper/2022/file/524ef58c2bd075775861234266e5e020-Paper-Conference.pdf ; https://github.com/tianzhou2011/FiLM |
| 8 | Ren et al., Time-Series Anomaly Detection Service at Microsoft (SR-CNN), KDD 2019 | Paper (ACM DOI + arXiv HTML) + MS docs | https://arxiv.org/abs/1906.03821 ; https://dl.acm.org/doi/10.1145/3292500.3330680 ; https://techcommunity.microsoft.com/blog/azuredevcommunityblog/overview-of-sr-cnn-algorithm-in-azure-anomaly-detector/982798 ; https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.timeseries.srcnnanomalyestimator |
| 9 | Yi et al., FreTS, NeurIPS 2023 | Paper (arXiv + NeurIPS PDF) + official repo | https://arxiv.org/abs/2311.06184 ; https://proceedings.nips.cc/paper_files/paper/2023/file/f1d16af76939f476b5f040fd1398c0a3-Paper-Conference.pdf ; https://github.com/aikunyi/FreTS |
| 10 | Yi et al., FourierGNN, NeurIPS 2023 | Paper (arXiv + NeurIPS PDF) + official repo | https://arxiv.org/abs/2311.06190 ; https://proceedings.neurips.cc/paper_files/paper/2023/file/dc1e32dd3eb381dbc71482f6a96cbf86-Paper-Conference.pdf ; https://github.com/aikunyi/FourierGNN |
| 11 | Zhang et al., TF-C, NeurIPS 2022 | Paper (arXiv + NeurIPS) + official repo + project page | https://arxiv.org/abs/2206.08496 ; https://proceedings.neurips.cc/paper_files/paper/2022/hash/194b8dac525581c346e30a2cebe9a369-Abstract.html ; https://github.com/mims-harvard/TFC-pretraining |
| 12 | Woo et al., CoST, ICLR 2022 | Paper (arXiv + author PDF) + official repo | https://arxiv.org/abs/2202.01575 ; http://www.mysmu.edu/faculty/akshatkumar/files/iclr22.pdf ; https://github.com/salesforce/CoST |
| 13 | Wang et al., FCVAE, WWW 2024 | Paper (arXiv + ACM DOI) + official repo | https://arxiv.org/abs/2402.02820 ; https://dl.acm.org/doi/10.1145/3589334.3645710 ; https://github.com/cstcloudops/fcvae |
| 14 | Zhou et al., TFAD, CIKM 2022 | Paper (arXiv) + official repo | https://arxiv.org/abs/2210.09693 ; https://github.com/DAMO-DI-ML/CIKM22-TFAD |
| 15 | Wei et al., "Benchmarking Inductive Biases for MTS-AD…", 2026 preprint | Paper (arXiv HTML, preprint) | https://arxiv.org/html/2605.28103 |
| 16 | FreCT: Frequency-augmented Convolutional Transformer (2025) | Paper (arXiv HTML, preprint, search-snapshot) | https://arxiv.org/html/2505.00941v1 |
| 17 | ATF-UAD, Neural Networks (2023) | Paper (ACM DL, search-snapshot) | https://dl.acm.org/doi/10.1016/j.neunet.2023.09.018 |
| 18 | TimeVQVAE-AD (2023) | Paper (arXiv PDF, search-snapshot) | https://arxiv.org/pdf/2311.12550 |
| 19 | Tuli et al., TranAD, VLDB 2022 | Paper (VLDB PDF) + official code (fetched) | https://vldb.org/pvldb/vol15/p1201-tuli.pdf ; https://github.com/imperial-qore/TranAD ; https://raw.githubusercontent.com/imperial-qore/TranAD/main/main.py ; https://raw.githubusercontent.com/imperial-qore/TranAD/main/src/models.py |
| 20 | "The Elephant in the Room: Towards A Reliable Time-Series Anomaly Detection Benchmark", NeurIPS 2024 D&B | Paper (NeurIPS PDF, search-snapshot) | https://proceedings.neurips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf |
| 21 | Local grounding | Docs/code (this repo) | `AGENTS.md`, `metrics/evaluate_utils.py`, `metrics/vus/utils/slidingWindows.py`, `metrics/vus/models/distance.py`, `RESEARCH_TS_AD_HEADS.md` |