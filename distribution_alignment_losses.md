# Distribution-Alignment Losses for TSAD / Time-Series SSL

Research date: 2026-09-03. Focus: what each divergence forces, collapse modes, when it helps anomaly detection.

## 1) KL in VAE ELBO — OmniAnomaly / InterFusion

**Primary:**
- Kingma & Welling, *Auto-Encoding Variational Bayes* — `arXiv:1312.6114` (VAE, ELBO, reparam trick).
- Su et al., *Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network (OmniAnomaly)* — KDD 2019, DOI `10.1145/3292500.3330672` — **no official arXiv** (proceedings + GitHub `NetManAIOps/OmniAnomaly`).
- Li et al., *Multivariate Time Series Anomaly Detection and Interpretation using Hierarchical Inter-Metric and Temporal Embedding (InterFusion)* — KDD 2021, DOI `10.1145/3447548.3467075` — **no official arXiv** (ACM DL + GitHub `zhhlee/InterFusion`).
- Baseline: Park et al., LSTM-VAE multimodal anomaly detector, IEEE RA-L 2018 (LSTM encoder + VAE reconstruction probability).

**Formula (ELBO):**
```
log p(x) >= E_{q_phi(z|x)}[log p_theta(x|z)] - KL(q_phi(z|x) || p(z)) := ELBO
Loss = -ELBO = L_recon + KL
Gaussian closed form: KL(N(mu,diag(sigma^2))||N(0,I)) = 0.5*sum(mu^2+sigma^2-1-log sigma^2)
```
- OmniAnomaly: sequential ELBO over t with GRU encoder/decoder + stochastic-variable connection `z_t|z_{t-1},h_t`, linear-Gaussian prior dynamics, planar flow for flexible `q`, reconstruction *probability* `E_q[log p(x|z)]` as score + POT threshold.
- InterFusion: hierarchical HVAE `q(z1,z2|x)=q(z1|x)q(z2|z1,x)` with `z1`=inter-metric, `z2`=temporal views, two-view + prefilter + RealNVP; ELBO has 2 KL terms (one per level).

**What it forces:** per-sample posterior near `N(0,I)`; compact/smooth latent; decoder must generate from small ball → normal manifold tight, anomaly = low `p(x|z)` / high recon + high KL.

**Collapse / failure:**
- Posterior collapse / KL-vanishing: `KL→0`, decoder ignores `z` (strong autoregressive decoder, too large KL weight, contaminated train). VAE then = deterministic AE with blurry mean predictions → misses subtle anomalies.
- Variance over-estimation (InfoVAE critique): ELBO inflates `q` variance to cover prior, smearing normal/anomaly boundary.
- Hierarchical addition failure: upper level collapses first if KL weights equal; needs KL annealing, free-bits, β<1 early, flows (both papers add flows for this reason).
- Scoring fragility: reconstruction-probability needsMonte-Carlo samples + POT tuning; point-adjusted F1 hides this.

**When helps TSAD:** noisy/stochastic telemetry (SMD/MSL/SMAP) where point error is noisy; need calibrated uncertainty + per-metric interpretability (reconstruction prob per variate); small clean-train where prior regularizes.

## 2) Anomaly Transformer — symmetric-KL Association Discrepancy + minimax

**Primary:**
- Xu et al., *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy* — ICLR 2022 Spotlight, `arXiv:2110.02642`.

**Formula:**
```
P^l = Softmax( Gaussian(|i-j|; sigma_i^l) )   # prior-association: local/adjacent bias, learnable sigma
S^l = Softmax( Q^l K^{lT} / sqrt(d) )          # series-association: data self-attention
AssDis_i(P,S;X) = (1/L) sum_l [ KL(P^l_{i,:}||S^l_{i,:}) + KL(S^l_{i,:}||P^l_{i,:}) ]  # symmetrized KL per point
Min-phase:  L = ||X-Xhat||_F^2 + lambda*||AssDis||_1  (fit P→S, stop-grad S)
Max-phase:  L = ||X-Xhat||_F^2 - lambda*||AssDis||_1  (push S away from P, stop-grad P)
Score_i = Softmax(-AssDis) ⊙ ||x_i-xhat_i||^2  # association-weighted recon; small AssDis = anomalous
```
Inductive claim: anomalies cannot build non-trivial global associations → `S≈P` (local) → small AssDis; normals spread globally → large AssDis.

**What it forces:** two-branch attention to disagree maximally under recon constraint; learnable `sigma` prevents fixed-window bias; multi-layer average gives multi-scale discrepancy usable directly as criterion.

**Collapse / failure:**
- `sigma→0` degeneracy: unconstrained max makes `P` a delta; *requires* alternating stop-grad + recon anchor, else meaningless.
- Attention collapse: uniform `S` (large AssDis everywhere) or identity `S` (zero AssDis); lambda-sensitive; O(N^2) window-size sensitive.
- Evaluation inflation: original SOTA relied on point-adjust; follow-ups show pure recon or learnable-prior ablations erase much of gap; weak on isolated point spikes vs contextual/segment anomalies it was built for.

**When helps TSAD:** contextual / shape / seasonality-break anomalies where point error alone ambiguous; periodic systems (SMD, PSM, SWaT) with long-range normal dependencies attention can exploit.

## 3) Alternatives: JS, MMD, Wasserstein

**Primary:**
- Goodfellow et al., *Generative Adversarial Nets* — `arXiv:1406.2661` (JS/minimax basis).
- Makhzani et al., *Adversarial Autoencoders* — `arXiv:1511.05644` (JS-adversarial latent matching).
- Gretton et al., *A Kernel Method for the Two-Sample Problem* — NeurIPS 2007 (MMD; no arXiv, JMLR version); Dziugaite et al., *Training Generative NNs via MMD Optimization* — `arXiv:1505.03906`.
- Zhao et al., *InfoVAE: Information Maximizing Variational Autoencoders* — `arXiv:1706.02262` (MMD/Stein/Adv vs ELBO; diagnoses variance over-estimation + uninformative code).
- Tolstikhin et al., *Wasserstein Auto-Encoders* — `arXiv:1711.01558` (WAE-MMD / WAE-GAN: aggregate-posterior matching).
- Arjovsky et al., *Wasserstein GAN* — `arXiv:1701.07875`; Gulrajani et al., *Improved Training of Wasserstein GANs (GP)* — `arXiv:1704.00028`.
- TSAD instantiations: MAD-GAN / BeatGAN / USAD (adversarial/JS-style recon+discriminator); WPS / WGAN-TSAD (W-distance + gradient penalty); WATCH `arXiv:2201.07125` (W-distance change-point).

**Formulas:**
```
JS(P||Q)  = 0.5*KL(P||M)+0.5*KL(Q||M), M=(P+Q)/2  in [0,log2], symmetric/bounded
MMD^2(P,Q)= E[k(z,z')]+E[k(w,w')]-2E[k(z,w)], k=RBF/imRQ;  =0 iff P=Q (characteristic k)
W_p^p(P,Q)= inf_{gamma in Pi(P,Q)} E_{(z,w)~gamma}[d(z,w)^p]; Gaussian W2^2=||m1-m2||^2+Tr(C1+C2-2(C1^{1/2}C2C1^{1/2})^{1/2})
WAE/InfoVAE: Loss = E[recon] + lambda*D( q_agg(z)||p(z) ), q_agg=E_x[q(z|x)], D=MMD/Adv/W
```

**What each forces:**
- JS/Adv: prior/posterior indistinguishable to discriminator; sharp samples, balanced overlap.
- MMD: match *all moments* of aggregate posterior, not per-sample → preserves `I(x;z)`, allows informative per-point `q(z|x)` while marginal stays prior-like.
- W2: minimal transport cost; geometry-aware, non-zero signal even with disjoint supports.

**Collapse / failure:**
- JS/Adv: saturates/vanishes when supports disjoint; mode collapse/drop; discriminator LR/schedule brittle; anomaly scores from `D` uncalibrated.
- MMD: RBF bandwidth-sensitive; O(n^2) (needs linear/RFF approx); too-weak kernel under-regularizes → anomalies also map in-prior; too-strong → same blur as KL.
- W: exact OT intractable → needs Kantorovich dual (WGAN critic + Lipschitz/GP) or Sinkhorn iterations; extra net + lambda tuning; contaminated train gets transported into prior too (overfits normal+anomaly together).

**When helps TSAD:**
- Use MMD/W over per-sample KL when: small train (InfoVAE/WAE show better sample efficiency + less variance blow-up), need sharp recon to catch subtle drifts, or normal/anomaly supports barely overlap (KL explodes/meaningless, W still orders by distance).
- Use JS/Adv when: need sharp generative scoring (MAD-GAN style) and can afford GAN stabilization; avoid as sole threshold — fuse with recon error.

## 4) Aligning sub-segments of same window to same latent distribution (view consistency)

**Primary:**
- Yue et al., *TS2Vec: Towards Universal Representation of Time Series* — AAAI 2022, `arXiv:2106.10466` (overlapping crops + timestamp masking, hierarchical temporal+instance InfoNCE on overlap).
- Tonekaboni et al., *Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding (TNC)* — `arXiv:2106.00750` (ADF-stationary neighborhood = positives, far = negatives, debiased contrast).
- Eldele et al., *Time-Series Representation Learning via Temporal and Contextual Contrasting (TS-TCC)* — `arXiv:2106.14112` (weak/strong aug, cross-view future prediction + contextual InfoNCE).
- Yang et al., *DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection* — KDD 2023, `arXiv:2306.10347` (in-patch vs patch-wise branches, symmetric KL + stop-grad, *no negatives*, pure discrepancy score).
- Oord et al., *Representation Learning with Contrastive Predictive Coding (CPC)* — `arXiv:1807.03748` (predictive-infoNCE ancestor).
- FreCT `arXiv:2505.00941` ablation: symmetric KL > asymmetric KL > JS for time/frequency dual-view consistency.

**Formulas:**
```
TS2Vec temporal: l_temp(i,t)=-log exp(r_{i,t}·r'_{i,t}) / sum_{t' in overlap}[exp(r·r'_{t'})+1_{t!=t'}exp(r·r_{t'})]
TS-TCC: L = L_TC(cross-view future InfoNCE) + L_CC(instance InfoNCE on contexts)
TNC: contrast N(neighbor window) vs Nbar(far window) with PU-debias for overlapping dynamics
DCdetector (closest to pure distribution-align): L = 0.5*KL(P||sg(N))+0.5*KL(N||sg(P)), P=in-patch, N=patch-wise; score=KL(P||N) per point
```

**What it forces:** same timestamp/segment in two contexts (different crop/mask/aug/permutation scale) → same embedding/distribution; normal dynamics = crop/permute-invariant; anomaly = view-disagreement. No decoder needed; learns stationary context at multiple scales.

**Collapse / failure:**
- Constant collapse: all views → same vector gives trivial zero loss. Cures: negatives (TS2Vec/TNC/TS-TCC InfoNCE), asymmetry+stop-grad (DCdetector BYOL/SimSiam-style), momentum/variance guards. Removing stop-grad in DCdetector/FreCT collapses.
- False negatives/positives: neighbor assumed positive contains same anomaly (miss); far assumed negative shares normal regime (hurts); augmentation too strong destroys signature, too weak trivializes.
- Nonstationarity violation: overlap-consistency assumes local stationarity; drifting normal flagged anomalous unless hierarchical/multi-scale pooling used (TS2Vec's fix).

**When helps TSAD:** pre-training on unlabeled nonstationary telemetry; contaminated train (contrastive less lured by large anomaly recon loss than AE/VAE); scoring by disagreement complements recon — best as `score = recon × f(view-KL)` (Anomaly-Transformer/DCdetector/FreCT pattern). Choose TS2Vec-style when need multi-scale timestamps; TNC when regimes switch slowly; DCdetector-style symmetric-KL+stop-grad when want negative-free, recon-free detector head.

### Rule of thumb for this repo (JEPA/conv-pyramid)
- Keep recon/prediction as primary; add *one* distribution aligner as regularizer, not replacement: per-sample KL → replace with aggregate MMD/W if posterior collapses or recon blurry; symmetric-KL between two subsegment views (DCdetector-style, with stop-grad) if need invariance to crop/phase without negatives.
- Always ablate `lambda=0` vs small lambda; monitor `KL/MMD` magnitude + active units + overlap-disagreement histogram on clean-train to catch collapse before trusting gains.
