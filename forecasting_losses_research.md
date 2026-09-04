# Forecasting / Prediction Losses for Time-Series Anomaly Detection

Research notes — Sept 2026. Focus: point-wise forecast errors vs shape-aware losses (Soft-DTW, DILATE/TDI).

## 1. MSE / MAE next-step forecasting (standard baseline)
- **Formula:** given window `x_{t-w:t}`, predict `x̂_{t+1}` (or horizon-k `ŷ_{1:k}`):
  `L_MSE = (1/N) Σ ||x_{t+1} − x̂_{t+1}||²`, `L_MAE = (1/N) Σ |x_{t+1} − x̂_{t+1}|`.
  Score at test: `s(t)=||x_t − x̂_t||²` (or MAE), threshold via POT / max-validation / epsilon-search.
- **Intuition:** learns conditional expectation of normal dynamics; anomaly = surprise. Sharp, localizable, cheap O(k·d).
- **Pros:** sensitive to spikes / point outliers; easy calibration; works streaming.
- **Cons:** penalizes small time-shift twice (predicts peak at t, truth at t+1 → double penalty); encourages blurry average forecasts on multi-step / non-stationary regimes; ignores shape.
- **Use in TSAD:** LSTM-ED / Telemanom (Hundman et al. 2018), GDN forecaster, MTAD-GAT forecaster all train on this.

## 2. MTAD-GAT joint forecast + reconstruction (Zhao et al. 2020)
- **Paper:** Zhao et al., *Multivariate Time-series Anomaly Detection via Graph Attention Network*, arXiv:2009.02040 (2020). Code: `ML4ITS/mtad-gat-pytorch`.
- **Loss:** `L = L_for + L_rec`
  - `L_for = (1/N) Σ ||x_{t+1} − x̂_{t+1}||²` — FC head on GRU+GAT features, RMSE.
  - `L_rec = −E_{q(z|X)}[log p(X|z)] + KL(q(z|X)||p(z))` — VAE ELBO reconstructing whole window X.
- **Inference score:** `Score(t) = Σ_i [(x̂_i−x_i)² + γ(1−p_i)]/(1+γ)` in paper; open implementations use `|pred−actual| + γ|recon−actual|` then mean over features, POT threshold.
- **Intuition:** complementary: forecast catches break of temporal pattern / contextual anomalies; VAE reconstruction catches distribution shift robust to noise. Ablation: either alone drops F1 significantly.
- **Tradeoff:** 2× heads + GAT cost; needs `γ` balance and POT tuning; VAE can still miss periodic-break anomalies where values stay in-range but order breaks.

## 3. GDN deviation scores (Deng & Hooi 2021)
- **Paper:** Deng & Hooi, *Graph Neural Network-Based Anomaly Detection in Multivariate Time Series*, AAAI 2021, arXiv:2106.06947. Code: `d-ailin/GDN`.
- **Train loss:** plain MSE forecasting with graph-attention over learned Top-K sensor graph: `L=||s^{(t)}−ŝ^{(t)}||²`.
- **Deviation scoring:**
  - `Err_i(t) = |s_i(t) − ŝ_i(t)|`  Eq.11
  - `a_i(t) = (Err_i(t) − median_i)/IQR_i`  Eq.12 — robust per-sensor z-score (median+IQR, not mean/std).
  - `A(t) = max_i a_i(t)`  Eq.13 — max over sensors (assumes few sensors hit). Threshold = max `A` on validation.
- **Intuition:** normalize away sensor scale so one noisy channel doesn't dominate; max preserves spike detectability; attention weights + predicted-vs-observed give root-cause explanation.
- **Tradeoff vs mean-fusion (MTAD-GAT):** max = high recall on single-sensor attacks (SWaT/WADI), but noisier; mean = smoother, better for diffuse anomalies.

## 4. Soft-DTW (Cuturi & Blondel 2017) — differentiable shape loss
- **Paper:** Cuturi & Blondel, *Soft-DTW: a Differentiable Loss Function for Time-Series*, ICML 2017, PMLR 70:894–903, arXiv:1703.01541. Code: `mblondel/soft-dtw`.
- **Background DTW:** `DTW(x,y)=min_{A∈A_{n,m}} ⟨A, Δ(x,y)⟩` where `Δ_{i,j}=δ(x_i,y_j)` (e.g. squared Euclidean), `A` binary warping path from (1,1) to (n,m) with →,↓,↘ steps.
- **Soft version:** replace hard `min` by soft-min:
  `min_γ(a)= −γ log Σ exp(−a_i/γ)` (γ>0), `min_0=min`
  `dtw_γ(x,y) := min_γ{⟨A,Δ⟩ : A∈A_{n,m}} = −γ log Σ_A exp(−⟨A,Δ⟩/γ)`
  - γ→0 recovers DTW; γ>0 = `−γ log K^γ_GA` (log Global-Alignment kernel).
- **Differentiability trick:** hard DTW gradient (when unique optimal path A*) is `∇_x DTW = (∂Δ/∂x)^T A*` — piecewise-constant, non-smooth. Soft-DTW gradient is expectation under Gibbs `p_γ(A)∝exp(−⟨A,Δ⟩/γ)`:
  `∇_x dtw_γ = (∂Δ/∂x)^T E_γ[A]`, `E_γ[A]=(1/Z) Σ_A A·exp(−⟨A,Δ⟩/γ)`.
  Computed by forward Bellman recursion + reverse backprop, no O(n²m²) enumeration.
- **Complexity:** forward O(n·m) time; full forward+backward O(n·m) time and space (classic DTW needs O(n·m) time, O(n) space if only value). Log-sum-exp stabilization required.
- **Intuition for TSAD:** shift/dilation-invariant: same spike 3 steps late ≈ small loss. Good for shape anomalies (missing peak, wrong morphology), bad if timing matters — motivates DILATE.
- **Weakness:** over-invariance (ignores when event happens); γ too large → over-smoothing / blurry barycenters; not a metric (triangle inequality fails, `dtw_γ(x,x)≠0` without debiasing/divergence correction).

## 5. DILATE — DIstortion Loss including shApe and TimE (Le Guen & Thome 2019/2022)
- **Papers:** Le Guen & Thome, *Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models*, NeurIPS 2019 pp.4191–4203, arXiv:1909.09020 (note: prompt cites arXiv:2010.08354 — that ID points to follow-up probabilistic STRIPE work; canonical DILATE is 1909.09020); journal extension *Deep Time Series Forecasting with Shape and Temporal Criteria*, IEEE TPAMI 2022, doi:10.1109/TPAMI.2022.3152862. Code: `vincent-leguen/DILATE`.
- **TDI origin:** Temporal Distortion Index, Frías-Paredes et al., *Energy* 94:180–194 (2016), doi:10.1016/j.energy.2015.10.093; Gastón et al., AIP Conf. 2017 — area between DTW optimal path and diagonal, i.e. horizontal (timing) vs vertical (amplitude) error; bi-criteria `(TDI, MAE)`.

### Formula
- Predict horizon-k `ŷ`, truth `y*`, cost matrix `Δ_{h,j}=δ(ŷ_h,y*_j)`, penalty matrix e.g. `Ω(h,j)=(h−j)²/k²` (quadratic; can be asymmetric to penalize late>early).
- `L_DILATE(ŷ,y*) = α·L_shape + (1−α)·L_temporal`, α∈[0,1]:
  - `L_shape(ŷ,y*) = DTW_γ = −γ log Σ_{A∈A_{k,k}} exp(−⟨A,Δ⟩/γ)`  — Soft-DTW above.
  - `L_temporal(ŷ,y*) = ⟨A*_γ, Ω⟩ = (1/Z) Σ_A ⟨A,Ω⟩·exp(−⟨A,Δ⟩/γ)`, `Z=Σ_A exp(−⟨A,Δ⟩/γ)`
    where `A*_γ = ∇_Δ DTW_γ = E_{p_γ}[A]` is soft optimal path (probability each cell is on path).
  - Hard TDI this smooths: `TDI=⟨A*,Ω⟩`, `A*=argmin_A ⟨A,Δ⟩`.
- **Tangled variant** `L_DILATE^t = −γ log Σ_A exp(−⟨A,αΔ+(1−α)Ω⟩/γ)` — puts Ω inside alignment search (generalizes Sakoe-Chiba band / Weighted DTW). Paper shows it disentangles worse than split DILATE because path itself is biased by Ω; split version penalizes timing of the *unconstrained* shape path.

### Differentiability / γ / complexity
- **Trick (two levels):** (1) soft-min for shape; (2) for time, can't soft-min directly (two matrices Δ vs Ω) → use identity `A*=∇_Δ DTW` and plug soft gradient `A*_γ=∇_Δ DTW_γ`. Then `L_temporal` is differentiable as expected TDI under Gibbs distribution. Custom DP forward+backward.
- **γ smoothing:** γ≈1e−2 typical. Small γ → near-hard DTW, sharp but bad local minima / unstable grads; large γ → smooth optimization, but washes out path, shape and time collapse toward mean alignment. Lipschitz grad ∝1/γ.
- **Complexity:** O(k²) time+space per series (k=horizon, e.g. 24–48); pairwise Δ dominates. Fine for scoring/forecast heads, heavy as per-step TSAD loss on long windows vs O(k) MSE.

### Strengths: spike vs shift anomalies
- **Spike / morphology anomaly:** `L_shape` fires when predicted shape can't warp to truth cheaply (missed drop, wrong amplitude, extra oscillation) even if MSE similar — Fig.1 NeurIPS: flat-line vs delayed-step have same MSE but very different DILATE.
- **Shift / delay anomaly:** pure Soft-DTW forgives delay; `L_temporal` explicitly charges `distance(path,diagonal)` — delayed step has small shape term but large temporal term. Lets detector separate "wrong shape" vs "right shape wrong time" (useful for ramp/energy, ECG, traffic).
- Empirically: DILATE-trained forecasters ≫ MSE-trained on DTW/TDI/ramp/Hausdorff metrics at comparable MSE; model-agnostic (FC, Seq2Seq-GRU, etc.).

### Weaknesses / caveats for anomaly detection
- Needs horizon-k forecast (not single-step); single spike in long horizon gets diluted by alignment.
- α sensitive (U-curve): α=1 → good shape, large time error; α→0 → time-only, MSE+shape explode (time meaningless without shape anchor). Must tune on validation (e.g. 0.5 synthetic/ECG, 0.8 traffic).
- Ω choice is prior (squared, asymmetric); wrong Ω misranks early vs late.
- Scale-mixing: shape (amplitude²) and time (steps²) units differ — α absorbs normalization, fragile across datasets.
- O(k²) + DP backprop slower, higher memory; log-sum-exp overflow if γ tiny.
- As *detection score* (not train loss): raw DILATE is less localizable than per-timestep |err| — need per-position attribution or combine `α·pointwise + (1−α)·DILATE`.

## 6. TDI vs MSE tradeoffs (takeaway)
- **MSE/MAE:** vertical-only, pointwise, O(n). Best for point/spike detection and thresholding; fails under timing jitter (double-penalty, blurry forecasts).
- **TDI (smooth):** horizontal-only, path-deviation, O(n·m). Best for latency/shift detection; meaningless alone (perfectly flat prediction can have zero TDI if path stays diagonal — must pair with shape).
- **DILATE = α·SoftDTW + (1−α)·SmoothTDI:** bi-criteria. Use as *training* loss for forecaster to get sharp + on-time normal model, but keep *detection* score as robust normalized residual (GDN-style median/IQR + max) or hybrid `score = pointwise + λ·DILATE_window`. Evaluate with triple `(MSE, DTW, TDI)` + ramp/Hausdorff, not MSE alone.

### Key citations
- Zhao et al. MTAD-GAT, arXiv:2009.02040 (2020).
- Deng & Hooi GDN, AAAI 2021, arXiv:2106.06947.
- Cuturi & Blondel Soft-DTW, ICML 2017, arXiv:1703.01541.
- Le Guen & Thome DILATE, NeurIPS 2019, arXiv:1909.09020; TPAMI 2022 doi:10.1109/TPAMI.2022.3152862.
- Frías-Paredes et al. TDI, Energy 2016 doi:10.1016/j.energy.2015.10.093; Gastón et al. 2017 doi:10.1063/1.4984517.
