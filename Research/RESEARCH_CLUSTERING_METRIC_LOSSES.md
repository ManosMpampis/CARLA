# Clustering & Metric Losses for Normal-Only TSAD (SMD-like)

Context: SMD = 28 machines × 38 metrics, 1-min, 5 weeks, 50/50 train/test split, ~4.16% test anomalies, per-machine training, train assumed normal-only [OmniAnomaly/Su et al. KDD19]. Need: normal-only, window/point scoring, no test-label tuning, temporal + cross-channel, multimodal normal regimes.
Saved 2026-09-03. Companion to `DESIGN_TSAD_JEPA.md`, `CONTEXT.md`.

## 1. Deep SVDD — One-class hypersphere
**Paper:** Ruff et al., *Deep One-Class Classification*, ICML 2018, PMLR 80:4393-4402. **No arXiv** (PMLR only: `proceedings.mlr.press/v80/ruff18a.html`). Do not confuse with `arXiv:1802.06360` (OC-NN, Chalapathy et al.).
**Loss intuition:** Minimum-volume sphere enclosing normals. Soft-boundary: `R^2 + 1/νn Σ max(0,||φ(x_i)-c||^2-R^2)+λ||W||^2`; One-class: `1/n Σ ||φ(x_i)-c||^2+λ||W||^2`. Score `s(x)=||φ(x)-c||^2`.
**Negatives/centers:** No negatives. One center `c` — fixed after AE-pretrain / initial forward mean, *not* learned.
**Collapse prevention:** `φ≡c` is trivial optimum. Must: remove biases, no bounded activations, fix `c`, weight decay, AE pretrain. Still fragile → DROCC, HSC, RPO variants.
**SMD suitability:** ★ Natural baseline. Per-machine one-class, window encoder (TCN) + distance score. Cheap, normal-only. Weakness: single sphere too coarse for multi-regime normal (idle/load/batch); consider multi-center.

## 2. Deep SAD — Semi-supervised inverse distance
**Paper:** Ruff et al., *Deep Semi-Supervised Anomaly Detection*, ICLR 2020. **arXiv:1906.02694**.
**Loss intuition:** Entropy view: low entropy for normal, high for anomaly. `L = unlabeled SVDD + η Σ_normal ||φ-c||^2 + η Σ_anom (||φ-c||^2+eps)^-1`. Labeled anomalies pushed out by *inverse* squared norm (bounded below, smooth — beats hinge/negative-L2 which diverges).
**Negatives/centers:** Same fixed `c` as SVDD. Labeled anomalies act as directed “negatives” but no contrastive pairs.
**Collapse prevention:** Same arch constraints as SVDD; inverse term actually helps spread.
**SMD suitability:** ★★ Only if you have labels. Pure normal-only → reduces to SVDD, no gain. Usable *without test leakage* by treating *synthetic injected* anomalies as labeled anomalies (`η=1`) — this is the CARLA/COCA trick. Don’t use test labels.

## 3. DAGMM — Deep Autoencoding GMM + energy
**Paper:** Zong et al., *Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection*, ICLR 2018. **No arXiv** (OpenReview `BJJLHbb0-`).
**Loss intuition:** `z=[z_c, d_rel(x,x'), d_cos(x,x')]` → estimation net `γ=softmax(MLP(z))` → differentiable GMM (means/covs via weighted MLE, no EM loop). `L = recon + λ1 E(z) + λ2 P(Σ)`, `E(z)=-log Σ_k π_k N(z|μ_k,Σ_k)`, `P` penalizes tiny diagonal cov. Score = energy `E(z)`.
**Negatives/centers:** No negatives. `K` learned mixture components (means+covs, typically K=2-4).
**Collapse prevention:** Recon term stops latent collapse; `P(Σ)` stops covariance singularity; joint end-to-end avoids decoupled AE-then-GMM optima. Still sensitive to K, λ, latent dim.
**SMD suitability:** ★★★ Best classical fit for multimodal normal. Captures load regimes as mixtures; recon+latent joint feature suits 38-D telemetry. Must wrap with temporal encoder (window → TCN/LSTM → z), not i.i.d. points. Tune per-machine.

## 4. DEC / IDEC — KL self-training clustering
**Paper DEC:** Xie et al., *Unsupervised Deep Embedding for Clustering Analysis*, ICML 2016. **arXiv:1511.06335**. **Paper IDEC:** Guo et al., *Improved DEC with Local Structure Preservation*, IJCAI 2017. **No arXiv** (IJCAI 1753-1759).
**Loss intuition:** Soft assign `q_ij ∝ (1+||z_i-μ_j||^2/α)^-(α+1)/2`, sharpened target `p_ij ∝ q_ij^2/f_j`. DEC: `L=KL(P||Q)`. IDEC: `L=L_recon+γ L_cluster (γ≈0.1)`.
**Negatives/centers:** No negatives. `K` centroids, k-means init after stacked-AE pretrain; `P` updated every T steps.
**Collapse prevention:** DEC drops decoder → feature distortion / degenerate clusters. IDEC keeps under-complete AE recon as local-structure preserver. Still needs pretrain, γ/λ coupling.
**SMD suitability:** ★ Poor as detector. Assumes K balanced clusters; tiny anomalies absorbed. Use only for regime discovery / pretrain regularizer, score by `max q` or dist-to-nearest-centroid if needed.

## 5. SwAV — Swapped cluster assignment
**Paper:** Caron et al., *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments*, NeurIPS 2020. **arXiv:2006.09882**.
**Loss intuition:** `L(z_t,z_s)=ℓ(z_t,q_s)+ℓ(z_s,q_t)`, `ℓ=-Σ q_s log softmax(z_t^T C/τ)`. Views predict each other’s *codes* `q` from Sinkhorn-Knopp optimal transport with equipartition. No pairwise comparisons, no memory bank; multi-crop.
**Negatives/centers:** No explicit negatives. `K` learned prototypes (e.g. 3000); uniformity constraint replaces negatives.
**Collapse prevention:** Equipartition + ε sharpening + prototype freezing; else uniform-assignment collapse.
**SMD suitability:** ★★ Pretrain only. Needs time-series-safe augmentations (mask/crop ok, strong jitter/scale risky). No native anomaly score (needs kNN/linear head). Prefer TS-native TS2Vec/TS-TCC over porting image SwAV.

## 6. ArcFace — Additive angular margin
**Paper:** Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019. **arXiv:1801.07698**.
**Loss intuition:** ℓ2-normalize `W,x`, `L=-log exp(s·cos(θ_y+m))/[exp(s·cos(θ_y+m))+Σ_{j≠y} exp(s·cos θ_j)]`. Geodesic margin `m` on hypersphere → intra-compact, inter-separated.
**Negatives/centers:** Needs class labels. Centers = class weights `W_j`; negatives = other classes in softmax. `s,m` hyperparameters.
**Collapse prevention:** Normalization + scale `s` stops norm collapse; margin stops class merge.
**SMD suitability:** ☆ Not for normal-only. Single class → no signal. Only usable with pseudo-classes (machine-ID, regime) + sub-center variant, then one-class head — semantics ≠ anomaly semantics. Skip as detector.

## 7. TS2Vec — Hierarchical temporal/instance contrast
**Paper:** Yue et al., *TS2Vec: Towards Universal Representation of Time Series*, AAAI 2022. **arXiv:2106.10466**.
**Loss intuition:** Dilated-CNN + timestamp-mask + random crop (contextual consistency). Dual InfoNCE at every max-pool scale: temporal (same time across 2 contexts = pos, other times = neg) + instance (same time same series across batch = pos, other series = neg). `L_hier=Σ_scales (L_temp+L_inst)`. Arbitrary subseries via pooling.
**Negatives/centers:** Yes — both axes, in-batch. No centers.
**Collapse prevention:** Negatives + hierarchy; robust to 50% missing; timestamp masking forces context use.
**SMD suitability:** ★★★ Strong encoder for SMD: multivariate, timestamp-level embeddings → per-step anomaly (mask-predict error / embedding distance), handles gaps. Caveat: stationary telemetry creates false negatives; add one-class/scoring head, don’t use raw contrastive score.

## 8. COCA — Contrastive one-class (sequence contrast + variance)
**Paper:** Wang et al., *Deep Contrastive One-Class Time Series Anomaly Detection*, SDM 2023. **arXiv:2207.01472**.
**Loss intuition:** Positive pair = (window repr `q`, Seq2Seq-reconstructed `q'`) — “sequence contrast”, no augmentation pairs. Invariance `d=mean[2-sim(q,Ce)-sim(q',Ce)]` (cosine to ℓ2-normalized center `Ce`, doubles as anomaly score 0-2) + variance `v=hinge(γ-std(q))` per-dim (VICReg-style). `L=d+λv(Q)+λv(Q')`.
**Negatives/centers:** No negatives (avoids pushing two normals apart — flaw of SimCLR-for-AD). One learned-normalized center `Ce`.
**Collapse prevention:** Variance hinge keeps batch std above threshold → solves hypersphere collapse without negatives or AE pretrain. Ablations NoVar collapses, NoOC/NoCL underperform.
**SMD suitability:** ★★★ Purpose-built for normal-only TSAD (NAB/AIOps/UCR/SMAP SOTA). LSTM+MLP projector fits SMD windows directly. Closest template for JEPA: replace Seq2Seq pair with context/target predictor pair, keep invariance+variance.

## 9. Triplet loss (FaceNet → CARLA pretext)
**Paper Triplet:** Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR 2015. **arXiv:1503.03832**. **Paper CARLA:** Darban et al., *CARLA: Self-supervised Contrastive Representation Learning for Time Series Anomaly Detection*, Pattern Recogn. 2025. **arXiv:2308.09296**.
**Loss intuition:** `L=Σ max(0,||f(a)-f(p)||^2-||f(a)-f(n)||^2+α)`. CARLA: `a=w_i`, `p=temporally-close w_{i-r}`, `n=injected-anomaly(w_i)` (point/contextual/collective). ResNet encoder + 2nd stage nearest/furthest-neighbour classifier. Rejects “aug=pos / distant=neg” assumption for series.
**Negatives/centers:** Yes — explicit negatives required + margin `α`. No center. Needs mining (hard/semi-hard, large batch).
**Collapse prevention:** `f≡const → L=α`, not minimum → no collapse if triplets non-trivial.
**SMD suitability:** ★★★ Best when synthetic faults available. Learns normal *and* deviation boundary → lower FPR than tight one-class; SOTA on 7 sets incl. SMD/MSL/SMAP. Risk: injection bias, temporal-closeness breaks under drift. Keep as historical pretext reference; reuse injection library for calibration probes.

### Takeaway for JEPA-TSAD (normal-only SMD)
- Detector: SVDD (baseline) < DAGMM / COCA / CARLA-triplet for multimodal/temporal normal.
- Encoder pretrain: TS2Vec hierarchy > SwAV; DEC/ArcFace not recommended.
- Anti-collapse menu maps directly: fix-center+bias-free (SVDD) vs covariance-reg (DAGMM) vs recon-preserve (IDEC) vs equipartition (SwAV) vs variance-hinge (COCA) vs margin+negatives (triplet).
- Semi-supervision without leakage: use Deep SAD form with *synthetic* anomalies only.

### Sources
- Ruff et al. ICML18 PMLR; Ruff et al. arXiv:1906.02694; Zong et al. ICLR18 OpenReview; Xie et al. arXiv:1511.06335; Guo et al. IJCAI17; Caron et al. arXiv:2006.09882; Deng et al. arXiv:1801.07698; Yue et al. arXiv:2106.10466; Wang et al. arXiv:2207.01472; Schroff et al. arXiv:1503.03832; Darban et al. arXiv:2308.09296; Su et al. OmniAnomaly KDD19 / SMD.
