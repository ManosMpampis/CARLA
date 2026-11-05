# Learning to compare a current window with its history

Research date: 2 November 2026. Scope: primary papers and author implementations; time-series anomaly detection is the intended application. This is a focused research survey, not an exhaustive literature review or an implementation specification.

## Finding

The closest match is **history-conditioned cross-attention trained with an explicit history–current compatibility objective**. These are two different design decisions:

1. **Mechanism:** current-window features query historical features, so historical content can affect the output.
2. **Objective:** train on correct versus mismatched history–current pairs, so ignoring one side makes the task harder.

Among the sources reviewed, **TranAD** is a direct architectural precedent; **Cross-RAG** is a direct historical-retrieval precedent; **CPC** and **TS-TCC** provide the most relevant predictive contrastive objectives. **DCdetector** is the strongest nearby example of training without reconstruction using attention-derived discrepancies, but its two views are not separate causal history and current windows. These fit assessments are our interpretation of the papers below.

None of these papers establishes that every change to every input must change every output. That requirement would also be undesirable for irrelevant noise. The useful requirement is: *changing relevant history, while fixing the current window, should change the prediction or compatibility score appropriately; changing relevant current evidence should do the same.* This is an experimental requirement, not something established by the existence of attention weights.

## Exact and adjacent precedents

### User-supplied candidate: AxonAD

AxonAD (*Surprised by Attention: Predictable Query Dynamics for Time Series Anomaly Detection*) is particularly relevant to the training-side requirement. Its abstract describes a reconstruction pathway coupled to a history-only predictor of future attention query vectors, trained with a masked prediction objective against an EMA target encoder. Scoring combines reconstruction error with recent-query cosine mismatch. That makes it a precedent for explicitly training predictability of attention representations, rather than assuming reconstruction will make them useful. It is not, from that description alone, proof of a current-Q/past-KV cross-attention mechanism or an exclusively attention-based loss. [Author abstract, September 2026 revision](https://arxiv.org/abs/2603.12916v4)

See the separate [AxonAD paper and code review](axonad_paper_code_review.md) for the detailed mechanism, implementation checks, and suitability assessment. This paragraph is based on the verified abstract; the separate review examines the full paper and code.

| Method | What interacts? | Training signal | Match to this question |
|---|---|---|---|
| TranAD, VLDB 2022 | Encoded local window queries sequence context in the paper | Reconstruction, focus-score self-conditioning, adversarial training | Closest anomaly-detection architecture |
| Cross-RAG, March 2026 preprint | Current query attends to retrieved historical inputs and their continuations | Forecast quantile regression in the reported setup | Closest retrieval architecture |
| CPC, 2018/2019 | Past context is compared with future latent features | InfoNCE | Closest foundational dependence objective; no required cross-attention |
| TS-TCC, IJCAI 2021 | Past context predicts future features across augmented views | Temporal and contextual contrasting | Time-series adaptation of predictive contrasting |
| DCdetector, KDD 2023 | Patch-wise and within-patch attention views of the same input | KL discrepancy with asymmetric/stop-gradient training | Attention-centered objective, different temporal setup |
| Anomaly Transformer, ICLR 2022 | Learned series association versus a temporal Gaussian prior | Reconstruction plus minimax association discrepancy | Attention regularization, not history–current cross-attention |
| MEMTO, NeurIPS 2023 | Encoded current input addresses learned normal prototypes | Reconstruction with memory training | Useful normal-memory comparison; memory is not the actual preceding history |

The following subsections supply primary evidence for the table.

### TranAD: current-window query, historical keys and values

The paper's context-based attention uses the encoded input window as queries and the encoded sequence as keys and values. It describes access to the sequence up to the current timestamp. Its two-stage reconstruction first produces a focus score from reconstruction error, then uses that score for self-conditioning; two decoders participate in adversarial training. This is not a loss exclusively on cross-attention weights. [Paper, §4.2–4.3](https://www.vldb.org/pvldb/vol15/p1201-tuli.pdf)

Implementation qualification: the current author code defines `TranAD.n_window = 10` and passes an encoded `src` as decoder memory. Consequently, do not infer unlimited historical access from the paper when reproducing the released code; inspect the actual data-window construction. [Author implementation, `TranAD`](https://github.com/imperial-qore/TranAD/blob/main/src/models.py)

**Assessment:** read first for the architecture, but its reconstruction task does not establish the stronger requirement that the model needs the correct history for every prediction.

### Cross-RAG: the current input chooses useful historical examples

Cross-RAG retrieves input–continuation pairs. Its cross-attention uses the current query representation as Q, retrieved input representations as K, and retrieved continuations as V. It also includes a query skip connection and a retrieval self-attention path. The backbone and predictor are frozen while added modules are trained; Appendix A.2 reports Chronos-Bolt's quantile regression loss. Thus, it is not an attention-only objective or a pure cross-attention-only network. It is a forecasting paper, not an anomaly detector, and the reviewed version is an arXiv preprint. [Paper, §3.3 and Appendix A.2](https://arxiv.org/html/2603.14709v1)

**Assessment:** especially relevant if “past” means a searchable bank of earlier normal episodes rather than only the immediately preceding crop. A causal anomaly detector must retrieve only information available before the scoring time; historical continuations must also have finished before that time.

### CPC: make history predictive of the next representation

CPC encodes observations as latents `z`, summarizes earlier latents into context `c_t`, and scores candidate future latents with a bilinear compatibility function. InfoNCE distinguishes the true future from negative candidates. The paper relates its ideal score to the ratio `p(future | context) / p(future)` and its objective to a mutual-information lower bound under the stated sampling setup. The original architecture uses autoregressive context and does not require cross-attention. [Paper, §2.2–2.3](https://arxiv.org/pdf/1807.03748)

In simplified notation:

```text
u(H, C) = z_Cᵀ W c_H / temperature
L = -log exp(u(H, C_positive)) / sum_j exp(u(H, C_j))
```

**Assessment:** this directly trains the relationship between past and current/future features. To prevent instance identity or machine identity from solving the task, candidate windows should include negatives from the same machine and comparable operating regime. Ordinary future-candidate contrasting alone is weaker than additionally fixing the current window and swapping its history; the latter is a proposed extension below.

### TS-TCC: predictive contrasting for time series

TS-TCC creates weak and strong augmentations, uses a temporal contrasting task to predict future features across views, and adds contextual contrasting to align contexts of the same sample and separate contexts of different samples. It is representation learning, not originally an anomaly scoring method. Cross-view prediction should not be confused with cross-attention between two feature banks. [Conference paper and abstract](https://www.ijcai.org/proceedings/2021/0324), [author code](https://github.com/emadeldeen24/TS-TCC)

**Assessment:** useful as a time-series objective template, but preserve anomaly-relevant details when choosing augmentations. An augmentation that erases an anomaly of interest would train the wrong invariance.

### DCdetector: attention-derived discrepancy without reconstruction

DCdetector forms patch-wise and in-patch attention views at multiple scales. It compares their representations with KL-based losses using asymmetric branches and stop-gradient operations. Its total objective is the signed combination `L_N - L_P`, rather than simply minimizing a symmetric KL term, and it has no reconstruction module. Its discrepancy produces pointwise anomaly scores. [Paper, §3.2–3.4, equations 8–11](https://arxiv.org/html/2306.10347)

**Assessment:** the closest match if “exclusively” means *learn by comparing attention-derived relations without reconstructing the signal*. It does not implement the requested split into causal past features and a separately encoded current window. Replacing its views with history and current requires a new, justified objective; its existing loss cannot simply be relabeled.

### Anomaly Transformer: regularize temporal associations

Anomaly Transformer contrasts a learned Gaussian temporal prior with series self-attention using symmetrized KL divergence. Its minimax scheme fits the prior to detached series associations and pushes series associations away from a detached prior under a reconstruction constraint. Its score combines association information and reconstruction error. This is within-sequence self-attention, not a separate past-to-current cross-attention module, and its loss includes reconstruction. [Paper, §3.1–3.2, equations 3–6](https://ise.thss.tsinghua.edu.cn/~mlong/doc/anomaly-transformer-iclr22.pdf)

**Assessment:** useful evidence that an objective can act on relationships rather than only signal error, but not the most direct starting point for the requested model.

### MEMTO: compare against learned normal memory

MEMTO uses a gated memory module containing prototypical normal features, two-phase training with K-means memory initialization, and an anomaly criterion combining deviations in input and latent space. [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256-Abstract.html)

**Assessment:** relevant if “history” means accumulated normal patterns. Learned prototype memory and the actual recent trajectory answer different questions: normal-pattern resemblance versus consistency with what this machine was just doing.

## Relation to the existing CARLA module

The inspected [local module](../../models/lewm_cross_attention.py) already implements much of the requested mechanism:

- `CrossAttentionLeWM._features` obtains context and second-crop features; `feature_extraction="separate"` isolates their feature extraction.
- In `query_source="observed"`, Q comes from the second crop, while K/V come from context. `CrossAttentionPredictor` has no query residual bypass.
- In `query_source="context"`, Q also comes from context, interpolated to the output length. This is a past-only forecasting arm, not the same conditioning task.
- Optional positional encodings enrich Q/K addressing. Prediction is produced from attended values through the reconstruction head.

These are code observations, not claims from the papers. The observed-query arm can use current information for addressing: it measures conditional reconstruction/compatibility after observing the current crop. It must not be described as an unseen-current forecast. Removing the query residual ensures that the output travels through the attention value path, but does not prove sample-specific dependence: constant values, uniform addressing, a head ignoring its input, or position-only matching remain possible behaviors.

## Proposed next method: two-sided compatibility training

The following is a **research proposal synthesized for CARLA**, not a published algorithm or a proven improvement. It extends the existing observed-query arm rather than claiming a new feature extractor is necessary.

Let `H_i` be historical features and `C_i` current-window features. Compute:

```text
A_ij = softmax(Q(C_i) K(H_j)ᵀ / sqrt(d))
R_ij = head(A_ij V(H_j))
e_ij = mean_abs(R_ij - stopgrad(target(C_i)))
u_ij = -e_ij / temperature
```

Use a fixed or frozen feature target initially (`target="features"` is a natural starting experiment), so the target cannot become trivially easy by co-adapting to the loss. This notation is token-major; the local implementation stores tensors channel-major.

For each fixed current crop, contrast its genuine history against alternative histories:

```text
L_history = -mean_i log exp(u_ii) / sum_j exp(u_ij)
```

For each fixed history, contrast its genuine continuation against alternative current crops:

```text
L_current = -mean_j log exp(u_jj) / sum_i exp(u_ij)
```

The two directions make “ignore history” and “ignore current” poor solutions when candidate sets truly require their relationship. Because the comparison already scores the cross-attention reconstruction, no auxiliary compatibility head can solve a separate task while the main output ignores history. However, the model could still exploit spurious identifiers or increase mismatched error through unhelpful behavior. Validate on unseen time spans and operating conditions.

Candidate experiments:

1. Existing reconstruction objective as baseline.
2. Reconstruction plus `lambda * (L_history + L_current)`.
3. Pure two-sided compatibility objective as the requested “comparison-only” experiment.

Option 3 requires special caution: relative ranking alone does not anchor the absolute scale or usefulness of reconstruction error for anomaly scoring. A correctly ranked positive pair need not have low absolute error. Keep it as an experiment, not the default recommendation.

Use negatives from non-overlapping periods of the same machine, matched for broad regime/scale where possible. Do not label every other normal continuation incompatible: periodic or stationary sequences can have several valid matches. Exclude near-duplicates, allow multiple positives, or use soft compatibility targets. Simple batch negatives from different machines may teach machine identification instead of dynamics. These sampling choices are proposed controls, not results from the surveyed papers.

If a loss operating directly on attention weights is mandatory, supervised/synthetic alignments could supply a target distribution over historical positions and cross-entropy could train attention against it. Natural telemetry usually lacks such alignment labels. Entropy penalties or imposing an arbitrary attention pattern are not substitutes for learning which history is useful. The useful-output objective should remain the main evidence of success.

## Verification of meaningful input dependence

Evaluate the following in `eval()` mode with fixed current crops, a held-out clean period, and deterministic inference. Measure both output changes and task/error changes; changing an output alone does not establish usefulness.

| Controlled intervention | What it tests |
|---|---|
| Hold C fixed; replace H with a plausible but incompatible historical segment | Whether correct historical content improves compatibility |
| Hold H fixed; swap C to a plausible incompatible continuation | Whether the score depends on the history–current relationship |
| Keep K/V fixed; replace Q with its mean or a learned constant | Whether current content controls retrieval |
| Keep Q fixed; replace V content with mean values | Whether retrieved historical content is used |
| Preserve positional encodings but swap content | Whether addressing is mostly positional |
| Remove attention and use a current-only head or historical mean | Whether cross-attention earns its added complexity |
| Add small irrelevant perturbations | Whether dependence is stable rather than noise sensitivity |

Jointly permuting historical K and V tokens without positional information does **not** change ordinary attention output: the corresponding softmax columns and values cancel the same permutation. Therefore, that permutation alone is not a valid test of learned temporal-order dependence. With positional encoding enabled, distinguish permuting already-positioned tokens from reassigning content to fixed positions.

Report `mean |R(H,C) - R(H_swapped,C)|`, positive-versus-mismatch score separation, and anomaly-detection metrics under each intervention. Use clean-train data for thresholds and training-time selection. Do not select losses, temperatures, history lengths, or negative samplers using test labels.

## Recommended reading and decision

Read **TranAD** for local-window/historical cross-attention, **CPC and TS-TCC** for the predictive contrasting objective, and **Cross-RAG** if historical retrieval is part of the intended design. Read **DCdetector** if eliminating reconstruction is the central requirement.

For this repository, the most informative next experiment is to keep the existing separate-feature, observed-query architecture and test whether the correct past improves the actual output. If it does not, add two-sided history–current compatibility training and repeat the same interventions. This recommendation is an inference from the mechanisms and local code, not a benchmark result or a claim that one surveyed method universally wins.
