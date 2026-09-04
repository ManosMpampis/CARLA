# RESEARCH: Output-Head + Loss Designs in Multivariate Time-Series Anomaly Detection

Date: 2026-08-23. Scope: research only. Every factual claim carries a primary-source citation (paper page +, where available, an official-code file opened during this session); anything not traceable to a primary source is labeled **Inference:** or **Speculative:** or excluded (see §6). All cited URLs were retrieved during this session — most by direct fetch, a few (flagged in §6/Sources) only via the session search tool's snapshot of the page.
Grounding for §5: local `AGENTS.md`, `WORKFLOW_PRETEXT.md`, and spot-reads of `carla_classification.py`, `models/models.py`, `losses/losses.py`.

---

## 1. TL;DR

- The owner's two-way hypothesis is **incomplete**: window/segment classification and reconstruction-error localization are the two largest clusters, but published MTS-AD heads also use at least five further mechanisms: **one-class boundary distance**, **explicit density (flow/GMM/diffusion)**, **critic realism**, **two-representation discrepancy**, and **forecast-error** scoring; plus **memory/prototype distance** as a reconstruction refinement.
- Verified exemplars per mechanism: window classification → CARLA (this repo's namesake paper) and COCA; boundary distance → Deep SVDD / Deep SAD; density → GANF (normalizing flow log-likelihood), DAGMM (GMM); critic → MAD-GAN's DR-score, TadGAN; discrepancy → Anomaly Transformer's association discrepancy and TS2Vec's masked-vs-unmasked embedding gap; forecasting → GDN's per-sensor deviation score; memory → MEMTO.
- Head+loss pairs are strongly coupled: the loss determines what quantity exists at inference (e.g., USAD's adversarial retraining term creates the second decoder whose error enters the score; DevNet trains a scalar score head *directly*, with no representation objective at all).
- Evaluation asymmetries are pervasive and leak-prone: verified examples include thresholds set with test-set statistics (Anomaly Transformer percentiles over train+test using the test anomaly ratio) and best-F1-on-test reporting before thresholding (OmniAnomaly pipeline steps).
- For THIS repo (pretext→classifier, i.e., family A): the cheapest verified additions are a **TS2Vec-style masked-view discrepancy score** (reuses the existing three-view encoder) and a **Deep-SVDD-style center-distance head** on frozen pretext embeddings. All §5 items are design inference.

---

## 2. Taxonomy of distinct head+loss mechanisms

| # | Mechanism (head ⇒ score) | How it differs from (a) window classification | How it differs from (b) reconstruction-error localization | Exemplar (verified) |
|---|---|---|---|---|
| A | Embedding → window label / neighbor-proximity | This IS (a) | No decoder, no input reconstruction | CARLA [§3.1], COCA [§3.2] |
| B | Decoder ⇒ log p(x\|z) "reconstruction probability" | Score is per-timestep, not per-window | This IS (b), but probabilistic (likelihood, not residual norm) | OmniAnomaly [§4.1] |
| C | AE(s) ⇒ residual norms (plain or adversarially retrained) | Per-timestep localization | This IS (b) in its plainest form | USAD [§4.2], MSCRED [§4.3] |
| D | Predictor ⇒ next-step forecast error | Per-timestep | Predicts unseen future values, not the observed window | GDN [§6.1], EncDec-AD/Hundman [§6.2] |
| E | Two parallel heads (forecast + recon) ⇒ weighted mix | Per-timestep | Mixes both error genera | MTAD-GAT [§6.3] |
| F | Two internal representations of the SAME series ⇒ KL/L2 discrepancy | Neither classifies windows nor measures input residuals | Discrepancy lives between distributions/embeddings, not x vs x̂ | Anomaly Transformer [§7.1], TS2Vec AD protocol [§7.2] |
| G | Density model ⇒ likelihood / sample energy | No classifier trained | No target output reconstructed (flow/diffusion denoise internally) | GANF [§8.1], ImDiffusion [§8.2], DAGMM [§8.3†] |
| H | Encoder ⇒ distance to learned center/boundary (or direct scalar score head) | Trained ON the AD objective, no pretext | Nothing reconstructed; geometry IS the score | Deep SVDD / Deep SAD / DevNet [§9] |
| I | Discriminator/critic ⇒ realism score (often ⊕ recon) | Per-window or per-segment | Only half the score (if combined) is reconstruction | MAD-GAN, TadGAN [§10] |
| J | Memory/prototype module ⇒ recon-from-prototypes + address distance | Per-timestep | Reconstruction constrained to normal prototypes | MEMTO [§4.4] |

† DAGMM's headline claims verified; its exact "sample energy" formula could NOT be re-verified this session (see §6).

---

## 3. Family A — Window/window-segment classification on learned representations

### 3.1 CARLA (= the method this repo implements)
- Venue/code: Darban et al., *Pattern Recognition* 157:110874 (2025); preprint [arXiv:2308.09296](https://arxiv.org/abs/2308.09296) (Related DOI confirms the journal version). Official code not located this session (see §6).
- Head: none in the classic sense — the "head" is the classification procedure over window representations. Paper: "We propose a self-supervised classification method that leverages the representations learned in the pretext stage to classify time series windows." ([arXiv HTML v3](https://arxiv.org/html/2308.09296v3))
- Scoring: proximity of neighbors in representation space — windows are classified "based on the proximity of their neighbours in the representation space", using nearest/furthest neighbors established at the end of the pretext stage (same source).
- Loss (stage 1): contrastive pretext with anomaly injection: "employs anomaly injection to learn similar representations for temporally proximate windows and distinct representations for windows and their equivalent anomalous windows"; injected negatives include point anomalies (spikes) and subsequence anomalies (seasonal/shapelet/trend). Entropy component coefficient ablated; 5 chosen as optimal (same source).
- Asymmetry: needs only unlabeled training windows; synthetic anomalies are manufactured, so no clean-only assumption beyond normality of the base stream; labels used only at evaluation.
- Local mirror: `carla_pretext.py` three-view triplet training and `carla_classification.py` self-supervised classification stage implement exactly this two-stage pattern (`WORKFLOW_PRETEXT.md`; `losses/losses.py:294-472` ClassificationLoss with pos/neg BCE terms).

### 3.2 COCA — contrastive coding + one-class objective fused into ONE stage
- Venue/code: SDM 2023 (pp. 694–702); official repo [ruiking04/COCA](https://github.com/ruiking04/COCA) (README fetched; paper behind SIAM paywall — link in README).
- Design claim (README abstract): "It treats the origin and reconstructed representations as the positive pair … namely 'sequence contrast'" and "'hypersphere collapse' is prevented by variance terms".
- Head: encoder f and a reconstruction branch producing a second embedding `feature_dec1` (`models/COCA/coca_trainer/trainer.py`).
- Scoring: `score = distance1 + distance_dec1` where each distance is `1 − cosine_similarity(feature, center)` to a hypersphere center `c` maintained as in Deep SVDD (`center_c` function in the same trainer).
- Loss: soft-boundary one-class term on the score (`length + (1/nu)*mean(relu(score − length))`) plus a variance regularizer penalizing feature std below 1: `sigma_loss = max(0, 1 − std)`; total `omega1*loss_oc + omega2*loss_sigma` (trainer.py).
- Asymmetry: single-stage (contrastive and one-class losses optimized together, per README critique of two-staged methods); evaluation uses Merlion RevisedPointAdjusted + affiliation metrics (trainer.py).
- Difference vs (a)/(b): the reconstruction branch exists only to make a *positive pair*; the score is boundary distance, not residual error.

### 3.3 TS2Vec downstream protocol (representation model, AD as application)
- Venue/code: AAAI 2022 ([arXiv:2106.10466](https://arxiv.org/abs/2106.10466)); official repo `zhihanyue/ts2vec`.
- Pretext loss: hierarchical contrastive — `hierarchical_contrastive_loss` sums instance-wise and temporal contrastive terms over max-pool hierarchy (`models/losses.py`).
- AD scoring (the interesting part): anomaly score = Σ|repr_masked − repr_unmasked| per timestamp, where repr_masked encodes with `mask='mask_last'`: `train_err = np.abs(all_train_repr_wom[k] - all_train_repr[k]).sum(axis=1)`; moving-average adjusted over a 21-step window; **threshold = mean(train_err_adj) + 4·std(train_err_adj)** computed on TRAIN errors only, plus delay-based post-processing (`tasks/anomaly_detection.py`).
- Asymmetry: fully unsupervised thresholding from train statistics (a good hygiene example).
- Note: paper abstract says only "we present a simple way to apply the learned representations for unsupervised anomaly detection" — the CODE reveals it is a masked-view discrepancy score (family F mechanism bolted onto a family-A model), not a classifier. Flagged in §11.

---

## 4. Family B/C/J — Reconstruction heads (probabilistic, deterministic-adversarial, memory)

### 4.1 OmniAnomaly — stochastic RNN + VAE + planar flows
- Venue/code: KDD 2019 (Su et al.); official repo [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) (README fetched; clone path in README points to the `smallcowbaby` mirror).
- Head: GRU-coupled VAE; posterior `RecurrentDistribution` with optional planar normalizing flows (`posterior_flow_type == 'nf'`); generator outputs Normal mean/std via dense layers (`omni_anomaly/model.py`).
- Scoring: exactly the Monte-Carlo reconstruction probability — docstring: "Get the reconstruction probability for `x`. The larger `reconstruction probability`, the less likely a point is anomaly." Implementation: `r_prob = p_net['x'].log_prob(...)` under z sampled n_z times from q (model.py `get_score`). So the score is a log-LIKELIHOOD, not an L2 residual.
- Loss: SGVB/ELBO: `loss = tf.reduce_mean(vi.training.sgvb())` over `log p(x|z)` + KL terms (model.py `get_training_loss`).
- Thresholding asymmetry: pipeline "Find[s] the best F1 score on the testing set" first, then "Init POT model on `train_score` to find the threshold" (README Processing section) — test-set F1 is consulted during threshold selection.
- Note: MC averaging over n_z samples of z is the "Monte Carlo reconstruction prob" the owner asked about — confirmed in code shape (`z_samples` mean/std across samples feed the score path).

### 4.2 USAD — two AEs + adversarial retraining
- Venue/code: KDD 2020 (Audibert et al.); official repo [manigalati/usad](https://github.com/manigalati/usad), all logic in `usad.py` (fetched).
- Head: shared Encoder + two Decoders (`decoder1`, `decoder2`), all MLPs.
- Loss (code, verbatim structure): `loss1 = 1/n·MSE(batch,w1) + (1−1/n)·MSE(batch,w3)`; `loss2 = 1/n·MSE(batch,w2) − (1−1/n)·MSE(batch,w3)` with `w3 = decoder2(encoder(w1))` — decoder2 is trained adversarially to reconstruct FAKE windows produced by encoder∘decoder1 (minimax sign flip).
- Scoring (code): `testing()` returns `alpha*mean((batch−w1)²,axis=1) + beta*mean((batch−w2)²,axis=1)` with defaults alpha=beta=0.5. So the score mixes the honest AE1 error with the adversarially-distorted AE2 error; α,β control sensitivity to false positives vs false alarms (paper framing; code confirms the formula and defaults).
- Asymmetry: trains on assumed-normal windows; n (epoch index) anneals the adversarial weight.

### 4.3 MSCRED — signature-matrix ConvLSTM autoencoder
- Venue/code: AAAI 2019 ([arXiv:1811.08055](https://arxiv.org/abs/1811.08055)); author demo repo [7fantasysz/MSCRED](https://github.com/7fantasysz/MSCRED) (minimal README; `code/` folder).
- Head: conv encoder → attention ConvLSTM → conv decoder reconstructing the input **signature matrices** (inter-sensor correlation images), not the raw series (abstract: "a convolutional decoder is used to reconstruct the input signature matrices").
- Scoring: "the anomaly score is defined as the number of poorly reconstructed pairwise correlations" — i.e., count of residual-signature entries above an empirically set θ (arXiv PDF experiments section, session-fetched excerpt). Root-cause = rows/columns of the residual matrix.
- Loss: square loss on signature-matrix reconstruction (paper methodology section, same source).
- Distinctive: reconstruction operates on a derived 2-D representation of correlations; scoring is COUNT-based (robust to scale, loses magnitude info).

### 4.4 MEMTO — memory-guided Transformer (multivariate; NeurIPS 2023)
- Venue: Song, Kim, Oh, Cho, NeurIPS 2023; arXiv [2312.02530]; project page github.com/gunny97/MEMTO (NeurIPS virtual page). *(Accessed via session search snapshots; see §6.)*
- Head: Transformer encoder + gated memory module storing prototypical normal patterns; decoding proceeds FROM the memory addresses, so "reconstructing abnormal samples using the stored features of normal patterns" yields normal-looking outputs (paper PDF, session search snapshot).
- Scoring: reconstruction error of the memory-decoded output (plus latent/memory-address distance in the paper's composite score — the composite detail not code-verified this session).
- Asymmetry: two-phase training with K-means initialization of memory items.
- Family J: reconstruction is bottlenecked through prototypes — closer to (b) than to anything else, but the memory-address divergence is an extra signal pure AEs lack.

### 4.5 InterFusion — hierarchical VAE (KDD 2021)
- Li, Zhao, Han, Su, Jiao, Wen, Pei, KDD '21, pp. 3220–3230; official repo [zhhlee/InterFusion](https://github.com/zhhlee/InterFusion). *(README/paper accessed via session search snapshots; see §6.)*
- Head: HVAE with two stochastic latents learning low-dimensional inter-metric (across features) and temporal embeddings; detection via MCMC-imputed reconstructions at anomalous segments (paper abstract/repo README).
- Scoring: VAE reconstruction-probability style pointwise scoring with the hierarchical latents enabling interpretation of point/context/subsequence anomaly types.
- Included here mainly to show family B extends naturally to structured latents.

---

## 5. (folded into families above) — see §4.

## 6. Family D/E — Forecasting and dual heads

### 6.1 GDN — graph deviation network (forecasting head, deviation-score normalization)
- Venue/code: AAAI 2021 ([arXiv:2106.06947](https://arxiv.org/abs/2106.06947)); official repo [d-ailin/GDN](https://github.com/d-ailin/GDN) (branch `main`).
- Head: learned sensor graph (top-k cosine of node embeddings) + GNN layer + per-node `OutLayer` MLP emitting ONE scalar per sensor: `out.view(-1, node_num)` (`models/GDN.py`). So the network predicts each sensor's value from its neighbors — a forecasting/regression head.
- Loss: MSE between predictions and ground truth (`get_loss = eval_mseloss`, `evaluate.py`).
- Scoring (the "deviation score", code): `err_scores = (test_delta − median_val_err) / (|IQR_val_err| + ε)`, smoothed over a trailing 4-step window; final score = sum of top-k sensors' normalized errors; threshold = **max of validation-set scores** (`get_err_scores`, `get_val_performance_data` in `evaluate.py`).
- Asymmetry: threshold anchored on validation max — uses labeled validation data implicitly for selection.

### 6.2 EncDec-AD / LSTM-NDT lineage (predict-then-threshold)
- EncDec-AD: Malhotra et al., ICML 2016 AD Workshop ([arXiv:1607.00148](https://arxiv.org/abs/1607.00148)): "learns to reconstruct 'normal' time-series behavior, and thereafter uses reconstruction error to detect anomalies" — technically a reconstructor, but the follow-up telemetry work made the predictive variant standard.
- Hundman et al., KDD 2018 ([arXiv:1802.04431](https://arxiv.org/abs/1802.04431)): LSTM prediction errors on spacecraft telemetry + "unsupervised and nonparametric anomaly thresholding" (per-channel smoothing, then quantile-of-|errors−μ| based dynamic threshold). Official code: khundman/telemanom (referenced by InterFusion baseline notes and OmniAnomaly data pipeline; repo itself not fetched this session).
- DeepAnT: Munir et al., IEEE Access 7:1991–2005, 2019 (author PDF at DFKI, metadata fetched: DOI 10.1109/ACCESS.2018.2886457). CNN "Time Series Predictor" module forecasts future values; anomaly when predicted vs actual deviate. *(Body text not machine-readable this session; scoring detail = Euclidean distance between predicted and actual window per common descriptions — treat as low-depth verification, §6.)*

### 6.3 MTAD-GAT — dual forecasting+reconstruction head
- Venue/code: ICDM 2020 ([abs](https://arxiv.org/abs/2009.02040); full text [HTML](https://arxiv.org/html/2009.02040)).
- Head: 1-D conv → parallel feature-oriented + time-oriented GAT layers → GRU → TWO heads: 3-layer FC forecaster (next timestamp) AND a VAE reconstructor (paper §III-A/III-D).
- Loss: joint `Loss = Loss_for + Loss_rec`, with RMSE for forecasting (Eq. 5) and VAE ELBO for reconstruction (Eq. 8) (full text).
- Scoring: Eq. 9: per-feature `s_i = [(x̂_i − x_i)² + γ·(1 − p_i)]/(1+γ)` summed over k features; threshold via Peak-Over-Threshold (POT); γ grid-searched on validation (γ=0.8 best).
- Code caveat (marketing-vs-code flag): the popular repo [ML4ITS/mtad-gat-pytorch](https://github.com/ML4ITS/mtad-gat-pytorch) explicitly deviates: "Instead of using a Variational Auto-Encoder (VAE) as the Reconstruction Model, we use a GRU-based decoder" and defaults `alpha=0.2` weighting forecast vs recon — numbers reproduced with THIS repo will differ from paper semantics.

---

## 7. Family F — Two-representation discrepancy heads (the owner's "discrepancy between two representations")

### 7.1 Anomaly Transformer — prior-association vs series-association discrepancy
- Venue/code: ICLR 2022 ([arXiv:2110.02642](https://arxiv.org/abs/2110.02642)); official repo [thuml/Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer) (`solver.py`, `model/AnomalyTransformer.py` fetched).
- Head: Transformer encoder whose Anomaly-Attention emits BOTH a learnable "series association" (softmax attention) and a Gaussian-kernel "prior association"; final `projection = nn.Linear(d_model, c_out)` reconstructs the input window.
- Loss (minimax, code): `loss1 = rec_loss − k·series_loss` (make series assoc far from prior) and `loss2 = rec_loss + k·prior_loss` (pull prior toward detached series), where each association term is a symmetric KL (`my_kl_loss`) between normalized prior and series distributions.
- Scoring (code): `metric = torch.softmax((-series_loss − prior_loss)/…·temperature)` with temperature 50, multiplied elementwise by the reconstruction MSE: `cri = metric * loss`. NOTE the negation: the code weights points where BOTH discrepancies are LOW — opposite in sign to the paper's "anomalies have high association discrepancy" narrative; combined with the min-max training, this remains a debated implementation choice (repo issue #14 referenced in code comments).
- Thresholding asymmetries (code): threshold = `np.percentile(combined_energy, 100 − anormly_ratio)` over TRAIN+TEST concatenated energies, with `anormly_ratio` taken from the TEST set's anomaly fraction; evaluation then applies point-adjust ("detection adjustment") inflating detections within ground-truth segments. Both inflate reported F1 relative to strict online use.
- Answer to owner: yes — this head produces neither a window label nor an input residual alone; it is the product of a cross-representation discrepancy and a residual, trained by minimax so the discrepancy becomes discriminative.

### 7.2 TS2Vec AD protocol — masked-vs-unmasked embedding gap
- See §3.3. Mechanically: score = ‖f(x with t masked) − f(x)‖₁ summed over dims. It requires NO decoder and NO labels — the "two representations of the original series" pattern the owner hypothesized, in its cheapest form.

---

## 8. Family G — Density / probabilistic heads

### 8.1 GANF — graph-augmented normalizing flows (ICLR 2022)
- [arXiv:2202.07857](https://arxiv.org/abs/2202.07857) (Comments field: "ICLR 2022. Code is available"); official code EnyanDai/GANF (linked from arXiv Comments; repo tree not opened this session).
- Hypothesis verbatim: "We hypothesize that anomalies occur in low density regions of a distribution and explore the use of normalizing flows for unsupervised anomaly detection".
- Head/score: flow density over each series conditioned on a learned Bayesian-network (DAG) over series; score = negative conditional log-likelihood; DAG and flow parameters jointly estimated. No reconstruction head exists.
- Loss: negative log-likelihood + DAG acyclicity regularization (paper).

### 8.2 ImDiffusion — imputation diffusion (VLDB 2023)
- [arXiv:2307.00754](https://arxiv.org/abs/2307.00754).
- Mechanism: diffusion model used as a time-series IMPUTER (masked segment recovered from neighbors). Key scoring sentence: "leverage the step-by-step denoised outputs generated during the inference process to serve as valuable signals for anomaly prediction".
- So the anomaly signal is the DENOISING TRAJECTORY (error across intermediate steps), not just a final reconstruction residual and not a raw likelihood — a genuinely distinct scoring surface inside the diffusion family.
- Official code: Microsoft (link on arXiv page not opened this session; see §6).

### 8.3 DAGMM — deep autoencoding GMM (ICLR 2018) — partially verified
- Canonical page: [OpenReview forum BJJLHbb0-](https://openreview.net/forum?id=BJJLHbb0-) — direct PDF fetch hit OpenReview's bot-check this session; UCSB author copy returned HTTP 403.
- Verified from the forum abstract (session search snapshot of that page): the autoencoder yields BOTH a low-dimensional representation and the reconstruction error, which are jointly fed to a GMM; autoencoder and mixture are optimized end-to-end with an estimation network instead of EM.
- **Not verified this session:** the exact "GMM sample energy" score formula. Widely stated, but per this document's rules it stays unclaimed here; community implementations (e.g., danieltan07/dagmm, mperezcarrasco/PyTorch-DAGMM) exist but are not author code.

---

## 9. Family H — Boundary / deviation heads (trained ON the AD objective)

### 9.1 Deep SVDD (ICML 2018)
- Paper page: [PMLR v80 ruff18a](https://proceedings.mlr.press/v80/ruff18a.html): the method "is trained on an anomaly detection based objective" (abstract) — explicitly contrasted with repurpose-the-pretext approaches.
- Official code: [lukasruff/Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch).
- Head: any encoder φ; score = squared distance to center c: `dist = torch.sum((outputs − self.c)**2, dim=1)`.
- Loss (code): one-class: `mean(dist)`; soft-boundary: `R² + (1/ν)·mean(relu(dist − R²))` with R updated as the (1−ν)-quantile of batch distances after warm-up (`optim/deepSVDD_trainer.py`).
- Asymmetry: center c initialized as mean of initial forward pass; AE pretraining optional; anti-collapse tricks required (bias-free nets / bounded-activation variants documented in repo networks).

### 9.2 Deep SAD (ICLR 2020) — adds labeled anomalies via inverse-distance penalty
- Paper: Ruff et al., ICLR 2020; arXiv [1906.02694]; official code [lukasruff/Deep-SAD-PyTorch](https://github.com/lukasruff/Deep-SAD-PyTorch).
- Loss (code, verbatim semantics): `losses = where(semi_target==0, dist, eta*(dist+eps)**semi_target)` — for KNOWN anomalies (semi_target=−1) the objective becomes η·(dist+eps)⁻¹, actively pushing anomalous embeddings AWAY from c; score at test = dist (`optim/DeepSAD_trainer.py`).
- Asymmetry: semi-supervised — needs a small pool of labeled anomalies; paper motivates via entropy of latent distributions (arXiv abstract).

### 9.3 DevNet (KDD 2019) — direct scalar score head
- Official code: [GuansongPang/deviation-network](https://github.com/GuansongPang/deviation-network) (repo + `devnet.py` fetched); arXiv [1911.08623].
- Head: MLP ending in `Dense(1, activation='linear', name='score')` — the OUTPUT IS THE ANOMALY SCORE; there is no representation-learning objective at all (README: "DevNet directly optimize[s] the anomaly scores").
- Loss (code): `deviation_loss` = z-score of the predicted score against a fixed Gaussian reference (5000 draws of N(0,1)): inliers minimize |dev|; outliers maximize `relu(margin − dev)` with `confidence_margin = 5.`; weighted by labels `(1−y)·inlier + y·outlier`.
- Asymmetry: weakly supervised — alternates inlier/outlier batches requiring a handful of LABELED anomalies; additionally injects noise outliers by feature-swapping (`inject_noise`).

---

## 10. Family I — Discriminator/critic-as-head

### 10.1 MAD-GAN (ICANN 2019)
- Paper: [arXiv:1901.04997](https://arxiv.org/abs/1901.04997); official code [LiDan456/MAD-GANs](https://github.com/LiDan456/MAD-GANs) (README + `AD.py`, `DR_discriminator.py` present; README notes RGAN base from ratschlab/RGAN).
- Head: LSTM-RNN generator reconstructs multivariate windows (rank-based reconstruction via search over candidates) and an LSTM-RNN discriminator.
- Scoring: "a novel anomaly score called DR-score to detect anomalies by discrimination and reconstruction" (paper abstract/session-fetched PDF) — discriminator output COMBINED with reconstruction residual.
- Asymmetry: GAN training instability acknowledged in the paper itself ("oscillations indicate the instability of GAN-based anomaly detection").

### 10.2 TadGAN (IEEE BigData 2020)
- Paper: [arXiv:2009.07769](https://arxiv.org/abs/2009.07769): cycle-consistency-trained LSTM generators (forward/backward mappings) with Critic(s); the authors "propose several novel methods to compute reconstruction errors, as well as different approaches to combine reconstruction errors and Critic outputs to compute anomaly scores" and report the best-suited combination per benchmark.
- Official implementation ships inside the authors' Orion benchmark (statement in abstract: "Our code is open source and is available as a benchmarking tool"); the specific scoring-function source file was not opened this session (see §6).
- Distinctive: interval-wise (not pointwise) processing via e.g. dtw/area-yielded recon-error variants pooled over intervals.

### 10.3 USAD's phase 2 as a degenerate critic — see §4.2: decoder2 trained to reproduce encoder∘decoder1 fakes is structurally the discriminator role, realized as an extra MSE term in the score.

---

## 11. Marketing-vs-code flags observed (each verified against the cited artifact)

| Model | Claimed | Observed in code/artifact |
|---|---|---|
| Anomaly Transformer | Association discrepancy separates anomalies | Test score = `softmax(−(series_KL+prior_KL)) · MSE` — discrepancy enters NEGATED, down-weighting high-discrepancy points; threshold uses TEST anomaly ratio; point-adjust applied (`solver.py`) |
| TS2Vec | "simple way to apply the learned representations for unsupervised AD" (abstract) | Protocol is a masked-view embedding discrepancy with train-statistics threshold (`tasks/anomaly_detection.py`) — a family-F mechanism, undocumented as such in the abstract |
| MTAD-GAT (popular PyTorch repo) | Presented as MTAD-GAT reproduction | Substitutes GRU decoder for the paper's VAE reconstructor; different α default (`ML4ITS/mtad-gat-pytorch` README) |
| OmniAnomaly | "uses the reconstruction probability to do anomaly judgment" (README) | Confirmed literally: log p(x\|z) via MC sampling — but it is a LIKELIHOOD, not a residual norm; pipelines also consult best-F1-on-test before POT thresholding (README Processing) |
| DevNet | Often described as "deep AD representation method" | Actually an END-TO-END SCORE learner: scalar head, no pretext, deviation loss vs Gaussian prior (`devnet.py`) |

---

## 12. Mapping to THIS repo (ALL of §12 is **Design inference**)

Repo facts used (`AGENTS.md`, `WORKFLOW_PRETEXT.md`, spot reads): pretext stage trains `ContrastiveModel` (backbone + projection head, `models/models.py`) with triplet-style `PretextLoss` over three views (`ts_org`, near view usually the preceding raw window, `ts_ss_augment` = `SubAnomaly`-injected anomaly); classification stage (`carla_classification.py`) runs `self_sup_classification_train` with `ClassificationLoss` computing pos/neg BCE over anchor/near/far similarities (`losses/losses.py:294-472`). This is precisely the CARLA paper's two-stage design (§3.1) → this repo sits in **family A**: it neither reconstructs inputs nor predicts futures anywhere in the pipeline.

### 12.1 What the taxonomy says about this position
- Family A models win on precision/generalization of *window-level* decisions but give **no timestep-localized score** without an auxiliary head. Every other family supplies one. If the owner wants localization, the verified literature says the minimal additions are (in rising cost): masked-view discrepancy (TS2Vec, §7.2) → center-distance head (Deep SVDD, §9.1) → latent-prediction score (JEPA line, see sibling doc `RESEARCH_JEPA_WORLD_MODELS.md`) → full reconstruction stack (USAD/OmniAnomaly, §4).
- The repo already owns the ingredients for the cheapest option: `SubAnomaly` injections and the three-view encoder are exactly what a discrepancy or deviation head consumes.

### 12.2 Candidate additions (ranked, all inference)
1. **Masked-view discrepancy scorer (TS2Vec-style, §7.2).** Freeze the pretext encoder; score(t) = Σ|E(x)[t] − E(mask_t(x))[t]|; threshold = train mean + 4σ (their hygiene). Reuses `AugmentedDataset` masking machinery; touches nothing in `PretextLoss` (keeps the `crop: True` crash dormant). New criterion name must be registered in the `get_criterion` whitelist (`AGENTS.md` trap).
2. **One-class head on frozen embeddings (Deep-SVDD-style, §9.1).** Compute center c once over pretext embeddings of training windows; report dist-to-c as a second score; optionally COCA's cosine variant + variance term (§3.2) since our projection head already L2-normalizes. Cheap; gives a window-level score complementary to k-means/silhouette checkpointing.
3. **Latent-prediction score (cross-reference).** The sibling research doc's Option A maps onto the near view ≈ preceding window; unchanged caveats there apply (collapse stabilization, device-default traps).
4. **Evaluation-hygiene alignment.** Adopt TS2Vec-style train-only thresholds; explicitly avoid the Anomaly-Transformer pattern (test-ratio percentiles, point-adjust) when reporting honest numbers — several verified SOTA pipelines embed these leaks (§7.1, §4.1).

### 12.3 Risks mapped to known traps (`AGENTS.md` / `WORKFLOW_PRETEXT.md`)
| Trap | Impact on any new head |
|---|---|
| `crop: True` crash (no `random_crop`) | Any masking must live outside `PretextLoss` |
| Criterion factory whitelist | Register new names; avoid legacy `criterion_kwargs` |
| Device default `cuda` in loss state | Create centers/masks on configured device |
| Train/eval mode mismatch on negative view | Apply uniform `.train()/.eval()` policy for paired-view scoring |
| O(N²) silhouette evaluation | Distance-head scores are O(N) — negligible addition |
| Hardcoded seed / worker seeds | Same reproducibility caveats for stochastic masking |

---

## 13. Limitations, dead ends, unverifiable

- **OpenReview bot-check:** `openreview.net/pdf?id=BJJLHbb0-` (DAGMM) served a CAPTCHA wall; UCSB author mirror returned HTTP 403. DAGMM's GMM-sample-energy score therefore remains **unclaimed** in this document (§8.3).
- **ScienceDirect blocks scripted clients:** CARLA journal page (HTTP 400) — journal identity instead verified via arXiv Related DOI + Monash/IBM records surfaced in session search.
- **Official code not located:** CARLA (Darban et al.) — no official repository found this session; the repo under study appears to be an independent implementation of the paper's design.
- **Accessed via session-search snapshots only (primary pages, not direct-fetch):** MEMTO NeurIPS/arXiv/project-page content (§4.4); InterFusion KDD'21/repo content (§4.5); DeepSAD arXiv abstract page (code file WAS fetched directly); MAD-GAN PDF excerpts; MSCRED PDF experiment section (scoring-count quote). Treat quoted wording from these as high-confidence but single-snapshot.
- **Not verified, excluded from claims:** D³R (diffusion decomposition; identified only via a secondary citation list); TimeGrad-derived AD scoring; TadGAN's Orion scoring source file (paper-abstract-level verification only); GDN-flow/other normalizing-flow detectors beyond GANF; DeepAnT's exact distance formula (metadata verified, body unreadable to the fetcher).
- **Wrong-ID traps caught during research (for the record):** arXiv 1806.09335 is NOT MAD-GAN (it is a blockchain paper; correct ID 1901.04997); arXiv 1809.04758 is GAN-AD (MAD-GAN precursor), not DeepAnT.

## 14. Sources (all accessed 2026-08-23)

| # | Source | Type | URL |
|---|---|---|---|
| 1 | Yue et al., TS2Vec, AAAI 2022 | Paper | https://arxiv.org/abs/2106.10466 ; code files https://raw.githubusercontent.com/zhihanyue/ts2vec/main/tasks/anomaly_detection.py , https://raw.githubusercontent.com/zhihanyue/ts2vec/main/models/losses.py |
| 2 | Wang et al., COCA, SDM 2023 | Repo (official) | https://github.com/ruiking04/COCA ; README https://raw.githubusercontent.com/ruiking04/COCA/main/README.md ; trainer https://raw.githubusercontent.com/ruiking04/COCA/main/models/COCA/coca_trainer/trainer.py |
| 3 | Darban et al., CARLA, Pattern Recognition 157:110874 (2025) | Paper | https://arxiv.org/abs/2308.09296 ; full text https://arxiv.org/html/2308.09296v3 |
| 4 | Su et al., OmniAnomaly, KDD 2019 | Repo (official) + code | https://github.com/NetManAIOps/OmniAnomaly ; https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/omni_anomaly/model.py ; https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/README.md |
| 5 | Audibert et al., USAD, KDD 2020 | Repo (official) + code | https://github.com/manigalati/usad ; https://raw.githubusercontent.com/manigalati/usad/master/usad.py |
| 6 | Zhang et al., MSCRED, AAAI 2019 | Paper + author demo repo | https://arxiv.org/abs/1811.08055 ; https://github.com/7fantasysz/MSCRED |
| 7 | Song et al., MEMTO, NeurIPS 2023 | Paper/project (search-snapshot) | https://neurips.cc/virtual/2023/poster/71519 ; https://arxiv.org/abs/2312.02530 ; https://github.com/gunny97/MEMTO |
| 8 | Li et al., InterFusion, KDD 2021 | Paper/repo (search-snapshot) | https://dl.acm.org/doi/10.1145/3447548.3467075 ; https://github.com/zhhlee/InterFusion |
| 9 | Deng & Hooi, GDN, AAAI 2021 | Paper + code | https://arxiv.org/abs/2106.06947 ; https://github.com/d-ailin/GDN ; https://raw.githubusercontent.com/d-ailin/GDN/main/models/GDN.py ; https://raw.githubusercontent.com/d-ailin/GDN/main/evaluate.py |
| 10 | Xu et al., Anomaly Transformer, ICLR 2022 | Paper + code | https://arxiv.org/abs/2110.02642 ; https://raw.githubusercontent.com/thuml/Anomaly-Transformer/main/solver.py ; https://raw.githubusercontent.com/thuml/Anomaly-Transformer/main/model/AnomalyTransformer.py |
| 11 | Zhao et al., MTAD-GAT, ICDM 2020 | Paper + popular repo | https://arxiv.org/abs/2009.02040 ; https://arxiv.org/html/2009.02040 ; https://github.com/ML4ITS/mtad-gat-pytorch |
| 12 | Malhotra et al., EncDec-AD, ICML'16 AD workshop | Paper | https://arxiv.org/abs/1607.00148 |
| 13 | Hundman et al., KDD 2018 telemetry | Paper | https://arxiv.org/abs/1802.04431 |
| 14 | Munir et al., DeepAnT, IEEE Access 2019 | Author PDF (metadata) | https://www.dfki.de/fileadmin/user_upload/import/10175_DeepAnt.pdf |
| 15 | Dai & Chen, GANF, ICLR 2022 | Paper | https://arxiv.org/abs/2202.07857 |
| 16 | Chen et al., ImDiffusion, VLDB 2023 | Paper | https://arxiv.org/abs/2307.00754 |
| 17 | Zong et al., DAGMM, ICLR 2018 | Paper (bot-blocked; see §13) | https://openreview.net/forum?id=BJJLHbb0- |
| 18 | Ruff et al., Deep SVDD, ICML 2018 | Paper + code | https://proceedings.mlr.press/v80/ruff18a.html ; https://raw.githubusercontent.com/lukasruff/Deep-SVDD-PyTorch/master/src/deepSVDD.py ; https://raw.githubusercontent.com/lukasruff/Deep-SVDD-PyTorch/master/src/optim/deepSVDD_trainer.py |
| 19 | Ruff et al., Deep SAD, ICLR 2020 | Paper + code | https://arxiv.org/abs/1906.02694 ; https://raw.githubusercontent.com/lukasruff/Deep-SAD-PyTorch/master/src/optim/DeepSAD_trainer.py |
| 20 | Pang et al., DevNet, KDD 2019 | Repo (official) + code | https://github.com/GuansongPang/deviation-network ; https://raw.githubusercontent.com/GuansongPang/deviation-network/master/devnet.py |
| 21 | Li et al., MAD-GAN, ICANN 2019 | Paper + repo (official) | https://arxiv.org/abs/1901.04997 ; https://github.com/LiDan456/MAD-GANs |
| 22 | Geiger et al., TadGAN, IEEE BigData 2020 | Paper | https://arxiv.org/abs/2009.07769 |
| 23 | Local grounding | Docs/code | `AGENTS.md`, `WORKFLOW_PRETEXT.md`, `RESEARCH_JEPA_WORLD_MODELS.md`, `carla_pretext.py`, `carla_classification.py`, `models/models.py`, `losses/losses.py` (this repo) |
