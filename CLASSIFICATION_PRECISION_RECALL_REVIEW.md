# CARLA Classification Precision/Recall Review

## Purpose

This document summarizes the current investigation into why classification results differ strongly by machine. For example, `machine-1-1` can have good metrics while `machine-2-1` can have very poor precision. It is intended as a handoff to another model for an independent review.

## Classification Design

CARLA does not train a conventional binary normal/anomaly classifier.

- The training set contains real normal windows only.
- Synthetic anomalies are generated from normal windows with `SubAnomaly`.
- The classification model has `num_classes: 10`, not two classes.
- Anchors and near neighbors are intended to form a normal/consistent group.
- Synthetic anomalies are intended to form a non-normal/inconsistent group.
- After training, the most frequent predicted class is defined as the normal class.
- Every prediction different from that class is treated as an anomaly.

Relevant code:

- Training objective: `losses/losses.py:294-488`
- Synthetic data and neighbor mining: `data/custom_dataset.py:96-261`
- Normal-class selection: `carla_classification.py:195-196`
- Anomaly scores and thresholding: `utils/evaluate_utils.py:568-592`

## Main Interpretation

The precision/recall imbalance can occur even when the representation is learning useful structure. The model is learning a multi-class self-supervised partition, while evaluation converts that partition into binary anomaly detection using a majority-class assumption.

This creates two common failure modes:

- **Low precision with reasonable recall:** multiple legitimate machine operating modes are assigned to non-majority classes, causing normal windows to be marked anomalous.
- **Low recall with reasonable precision:** real anomalies are mapped to the normal class because they differ from the synthetic anomalies used during training, or the anomaly threshold is too conservative.

## Finding 1: Configuration Changes the Objective

The classification experiment files do not all train the same loss.

### `configs/classification/experiments/original.yml`

```yaml
entropy_weight: -2.0
inconsistency_weight: 0
consistency_weight: 1.0
classification_loss_flag: False
```

With the current entropy implementation, entropy is positive:

```python
entropy_loss = positive_entropy - negative_entropy
marginal_total_loss = consistency_loss + entropy_weight * entropy_loss
```

Therefore, `entropy_weight: -2.0` encourages high anchor entropy rather than confident class collapse. Inconsistency is disabled, and the classification mixing term is disabled. Synthetic negatives receive no effective separation objective in this configuration.

This configuration can produce arbitrary class assignments, poor anomaly recall, or many false positives after the majority-class conversion.

References:

- `configs/classification/experiments/original.yml:7-15`
- `losses/losses.py:420-467`

### `configs/classification/experiments/base.yml`

```yaml
entropy_weight: 2.0
inconsistency_weight: 1.3
consistency_weight: 1.0
classification_loss_flag: False
```

This configuration has a positive entropy weight and an active anchor-negative inconsistency term. It is more aligned with separating synthetic negatives, but it still has no fixed binary normal/anomaly target.

The original.yml file is not used and should be not take into account.

### Classification-enabled experiment configurations

Some configs set `classification_loss_flag: True`, for example:

- `configs/classification/experiments/classification.yml`
- `configs/classification/experiments/disimilarity-classification.yml`
- `configs/classification/experiments/entropy_all-classification.yml`
- `configs/classification/experiments/entropy_all-disimilarity-classification.yml`

The loss computes:

```python
shift_weight = (
    (anchors_prob.std() - negatives_prob.std())
    * (n / torch.sqrt(n - 1))
    * classification_loss_flag
)
total_loss = (
    (1 - shift_weight) * marginal_total_loss
    + shift_weight * classification_loss
)
```

`shift_weight` is not clamped. It can be below `0` or above `1`, which can reverse the contribution of one loss component. This behavior was identified but intentionally not changed yet. It should be logged and checked before comparing machines.

Reference: `losses/losses.py:462-467`.

## Finding 2: Normal-Class Selection Can Create False Positives

`DynamicNeighbors.predict_and_update()` returns predictions for three groups:

1. Original anchors
2. Weak/near views
3. Synthetic anomalies

The classification loop combines all predictions before selecting the normal class:

```python
label_counts = torch.bincount(predictions["predictions"])
normal_label = label_counts.argmax()
```

References:

- `data/custom_dataset.py:131-158`
- `carla_classification.py:195-196`

If normal windows occupy several classes, the majority class may represent only one operating mode. Other valid normal modes become false positives.

The synthetic anomaly group can also influence the selected mode. This is considered intentional because real anomalies are unavailable in training, but it must be measured when diagnosing precision problems.

## Finding 3: Hard Mining Is Machine-Dependent

Neighbor mining is intentionally difficult:

- For the weak-neighbor pool, the largest feature distances are selected.
- For the synthetic-anomaly pool, the smallest feature distances are selected.
- One candidate is sampled from each stored top-k group per training item.

References:

- `data/custom_dataset.py:160-174`
- `data/custom_dataset.py:245-253`

This means the classification loss is trained on:

- A difficult positive that may belong to a different normal operating regime.
- A difficult synthetic anomaly that may look almost normal.

With `update_data: false`, these pairs are mined once from the initial representation and stay fixed while the classifier changes. This is an accepted design option, but it can create different behavior across machines.

## Finding 4: Synthetic and Real Anomalies May Differ

The only anomaly-like samples used during training are produced by `SubAnomaly`.

Real validation anomalies are never used in the classification loss. A machine can therefore have:

- Good synthetic-anomaly separation but poor real-anomaly recall.
- Real anomalies that look normal in the learned output space.
- Real normal operating modes that look more like synthetic anomalies, reducing precision.

The configured `portion` value is currently not used to control the injected subsequence length. `SubAnomaly` randomly selects approximately 10% to 90% of a window.

References:

- `data/augment.py:94-103`
- `utils/common_config.py:269`

## Finding 5: Train/Evaluation Mode Asymmetry

For full-network classification training:

- Anchors run with `model.train()`.
- Near neighbors run with training behavior.
- Far neighbors run after `model.eval()`.

References:

- `utils/train_utils.py:87-127`

With BatchNorm and dropout, the negative branch uses different statistics and stochastic behavior from the anchor and positive branches. This can create unstable machine-specific decision boundaries.

## Finding 6: Thresholds and Metrics Have Different Semantics

The code reports several different evaluation concepts:

- Direct classification: `argmax(output) != normal_label`.
- Anomaly score: `1 - p(normal_label)`.
- Validation-best threshold: optimized using validation labels.
- Training-derived threshold: optimized using synthetic anomaly labels.

References:

- `utils/evaluate_utils.py:568-592`
- `carla_classification.py:198-208`

The training-derived threshold is calibrated on synthetic anomaly prevalence, not the real machine anomaly prevalence. A machine with rare real anomalies can show very poor precision because a small false-positive rate dominates:

```text
precision = TP / (TP + FP)
```

The validation-best threshold may show better results because it is optimized directly on the validation labels. It should be treated as a tuning result, not an unbiased final test result.

## Finding 7: Overlapping Windows Amplify Errors

SMD uses windows with approximately:

- `wsz = 256`
- `stride = 5`

References:

- `data/original_dataset.py:29-70`
- Active classification configs such as `configs/classification/experiments/base.yml:31-38`

A single window-level false positive can affect many overlapping time points. This can produce a large point-level false-positive region and sharply reduce precision.

Similarly, a short anomaly must influence enough windows to be detected reliably. Otherwise recall can be low even if the window classifier is partially correct.

## Precision/Recall Diagnostic Interpretation

### Low precision with reasonable recall

Check:

- Whether the normal class is correct.
- Whether original validation normals occupy multiple predicted classes.
- Whether the anomaly prediction rate is too high.
- Whether the selected threshold is too low.
- Whether a small false-positive window error expands through overlap.
- Whether `normal_label` changes between epochs or checkpoints.

### Low recall with reasonable precision

Check:

- Whether real anomalies differ from synthetic anomalies.
- Whether the selected threshold is too high.
- Whether the model maps real anomalies to the normal class.
- Whether the active config disables inconsistency or negative learning.
- Whether hard synthetic negatives are too difficult or stale.
- Whether anomalies are too short relative to the window/stride setup.

### Both precision and recall low

Check:

- Whether the representation separates normal and synthetic views at all.
- Whether the normal label is unstable.
- Whether the loss is finite and has meaningful gradients.
- Whether the active config has an unintended entropy sign or disabled terms.
- Whether the selected checkpoint is actually the intended model artifact.

## Required Machine-Level Diagnostics

Collect these values for `machine-1-1` and `machine-2-1` at initialization, epoch 0, the selected checkpoint, and the final checkpoint.

1. Predicted class histograms for original anchors, weak views, synthetic anomalies, validation normals, and validation anomalies.
2. `normal_label` computed from anchors only, anchors plus weak views, and all three training groups.
3. Quantiles of `p(normal_label)` for validation normals and validation anomalies.
4. Anomaly rate under direct argmax classification, training-derived threshold, and validation-best threshold.
5. `TP`, `FP`, `FN`, `TN`, FPR, FNR, anomaly prevalence, event count, and average event length.
6. `consistency_loss`, `inconsistency_loss`, `positive_entropy`, `negative_entropy`, `classification_loss`, `marginal_total_loss`, and `total_loss`.
7. Minimum, median, maximum, and out-of-range fraction of `shift_weight`.
8. Anchor-positive and anchor-negative feature distances before and after classification training.
9. Separate gradient norms for the backbone and cluster head.
10. Difference between repeated train-mode and eval-mode predictions on the same machine batch.

## Independent Review Request

Review this summary against the source code and answer:

1. Which findings are definite implementation bugs versus intended behavior?
2. Which active configuration is most likely responsible for the observed machine-specific precision/recall behavior?
3. Does the loss provide enough direct pressure to separate real anomalies, given that only synthetic anomalies are available during training?
4. Is the majority predicted class a reliable normal-label estimator when there are ten output classes?
5. Which diagnostics should be added first to distinguish normal-mode false positives from synthetic-to-real anomaly mismatch?
6. Should future changes target the loss, synthetic anomaly generation, normal-label selection, threshold calibration, or evaluation reconstruction first?

No conclusions should be drawn about `machine-1-1` versus `machine-2-1` without their actual prediction distributions and checkpoint logs.
