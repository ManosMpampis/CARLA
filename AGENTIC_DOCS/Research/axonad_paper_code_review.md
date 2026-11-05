# AxonAD: fit for comparing history with the current observation

Reviewed 2026-11-02. Research and source inspection only; no CARLA model changes.

## Verdict

**Strong candidate for the history-versus-current comparison objective. Partial fit for an architecture where current features explicitly retrieve past features through cross-attention.** These are two separate experimental questions.

The paper is *Surprised by Attention: Predictable Query Dynamics for Time Series Anomaly Detection*, arXiv:2603.12916, reviewed at **v4**. It proposes predicting attention queries from past context and comparing them with EMA target queries, alongside reconstruction. [Paper, §§3.1–3.4](https://arxiv.org/html/2603.12916v4)

Official code: [iis-esslingen/AxonAD](https://github.com/iis-esslingen/AxonAD), downloaded for inspection to `/tmp/carla-axonad-review`. Inspected commit: `0d7cde688b1eaf0e6230fc4f441623da9c5c615b`. All code references below are pinned to that commit. No full training or benchmark reproduction was performed.

## What the released model actually computes

The following is a source-code reading and mathematical restatement, rather than a claim of reproduced detection performance. [Core implementation](https://github.com/iis-esslingen/AxonAD/blob/0d7cde688b1eaf0e6230fc4f441623da9c5c615b/TSB_AD/models/AxonAD.py)

```text
Past input through t-s → embedding → shift → causal TCN → predicted query q̂_t
Observed input at t   → EMA embedding + query projection → target query q̄_t
                                                        ↓
                                      mismatch = 1 − cosine(q̂_t, q̄_t)

Full observed window → self-attention + residual + feedforward → reconstruction
```

- `TCNTransformerModel.forward`, lines 270–310: the online representation is a per-timestep linear embedding plus a learned positional vector, followed by LayerNorm. The target representation is its frozen EMA counterpart. The target query is projected **before attention**; it is not an attention weight map or a context-enriched attention output.
- `PredictiveAttention.forward`, lines 90–128: a shift by `forecast_steps` excludes current/future inputs from each query prediction when that setting is positive. A causal convolutional network forecasts queries. Reconstruction instead uses observed Q/K/V with bidirectional self-attention, `is_causal=False`.
- `TCNQueryPredictor`, lines 37–59: four kernel-3 layers have dilations 1, 2, 4, 8. The receptive field is `1 + 2*(1+2+4+8) = 31` shifted input positions. Away from boundaries, prediction at t depends on raw positions `[t-s-30, t-s]`, rather than the whole window or an unlimited history. This span is derived from the convolution geometry.
- The EMA target query at t depends on the channel vector at t and its window-relative position. It has no temporal mixing in this implementation. Thus this is **past context versus current projected features**, rather than two independently encoded temporal windows.
- Predicted queries are returned for supervision/scoring, but are not used in reconstruction attention. Shared parameters couple training; at fixed weights there is no predictor-to-reconstruction forward path.

## Loss and score

From `_jepa_masked_loss`, lines 446–471, and `fit`, lines 538–554:

\[
L_q=\operatorname{mean}_{b,h,t\in M}\left[1-\cos(\hat q_{b,h,t},\operatorname{sg}(\bar q_{b,h,t}))\right],
\]

\[
L=e^{-a}L_{\rm MSE}+a+e^{-b}L_q+b.
\]

The learned scalars balance reconstruction and prediction. The target has no gradient; the online embedding and predictor can receive prediction gradients. EMA updates occur after optimizer steps. The mask selects loss positions; it does not erase the input values. Mask sampling focuses on a tail region based on `(T-s)//2`, whereas deployed mismatch aggregation uses `kl_tail_k`.

From `_calibrate` and `decision_function`, lines 480–655: training-window medians/IQRs normalize reconstruction error and query mismatch, and the two normalized scores are added. A KL attention diagnostic is computed but excluded from the default sum. Despite the variable name `s_kl`, the default training term is cosine query prediction, not attention-map KL. [Pinned loss and scoring source](https://github.com/iis-esslingen/AxonAD/blob/0d7cde688b1eaf0e6230fc4f441623da9c5c615b/TSB_AD/models/AxonAD.py#L424)

## Does it deliver the requested input dependence?

**Yes as a mechanism:** the predictor has a path from recent history; the target has a path from the observed current input; their disagreement enters the deployed score. This is more directly aligned with the requested comparison than simply attaching attention to a reconstruction model.

**No universal guarantee:** neither a nonzero computational path nor low cosine prediction loss proves that a trained network uses informative history on every example. Constant or position-dominated query representations remain a concern to measure. EMA stabilizes targets but does not mathematically enforce a noncollapsed representation. Also, cosine distance ignores query magnitude, and query direction differences do not uniquely establish changed attention distributions: routing depends jointly on queries and keys. These are analytical limitations, not demonstrated defects of a trained AxonAD checkpoint.

The desired property should be *useful dependence*: irrelevant changes to old history should not necessarily alter the output. Requiring a different answer after every input change would conflict with robustness.

## Small executable checks performed

Used CARLA's `./venv/bin/python`, imported the official model from the temporary checkout, and ran CPU checks with seed 4, batch 2, window 64, channels 3, embedding 16, heads 4, horizon 2. Model was untrained and in eval mode. Examined zero-based position t=50.

| Intervention | Maximum absolute difference | Meaning |
|---|---:|---|
| Change inputs at t and later; inspect predicted query at t | 0 | No current/future leakage in predictor for this configuration |
| Same intervention; inspect target query at t | 1.379688 | Target responds to observed current input |
| Change recent past `[t-4:t-1)`; inspect predicted query at t | 0.013327 | A history-to-prediction path exists |
| Change remote past `[:t-32)`; inspect predicted query at t | 0 | Consistent with the finite receptive field |
| Change recent past only; inspect target query at t | 0 | Target is pointwise in time |
| Zero predictor parameters; compare reconstruction | 0 | Predicted queries do not directly drive reconstruction |

Assertions passed. These checks verify computation paths only; they do not establish trained dependence, anomaly sensitivity, or benchmark quality.

## Suggested CARLA experiment — synthesis, not the published AxonAD model

The local [`models/lewm_cross_attention.py`](../../models/lewm_cross_attention.py) already supports observed queries with past keys/values, and removes a query residual bypass. Its `query_source=context` alternative is past-only forecasting; `feature_extraction=separate` isolates crop receptive fields. Source inspected from the current working tree, which already contains user modifications.

Two coherent experiments answer different questions:

1. **Predict expected current queries from history.** Preserve an observed query stream; predict its clean frozen/EMA target from a separate past crop. Add cosine mismatch as a score alongside the existing residual. This tests the central AxonAD idea. Choose the history encoder/receptive field deliberately, and avoid targets that can shrink or become constant while the predictor follows them.
2. **Require current-to-past retrieval.** Keep `Q=current, K/V=past`, then compare the history-supported output with clean current features. An auxiliary expected-query loss can complement this retrieval objective. This combined design is an adaptation, not an AxonAD reproduction, and an auxiliary loss alone does not force the retrieval output to use history.

Before choosing either, train clean chronological splits and compare: residual only, query mismatch only, combined score, and a current-only control. At fixed current input, replace history with plausible history from an incompatible regime; compare true versus substituted histories and temporal-order perturbations. Test the reverse intervention with history fixed and current changed. Include both amplitude anomalies and coupling/delay anomalies, inspect query variance and positional shortcuts, and report changes in detection quality as well as numerical output changes.

For CARLA integration, retain per-timestep mismatch before aggregation and use the repo's cover-count-aware scorer. The official detector creates window scores and center-pads them to input length, even though the query term emphasizes the tail; copying that alignment would need separate validation. Use clean-train calibration and keep test labels out of all tuning. [Official output alignment](https://github.com/iis-esslingen/AxonAD/blob/0d7cde688b1eaf0e6230fc4f441623da9c5c615b/TSB_AD/models/AxonAD.py#L613)

Recommendation: use AxonAD as the most directly relevant **query-predictability baseline** in this investigation, and treat explicit current-to-past cross-attention as a separate architectural choice. Success for one does not establish success for the other.
