from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from torchmetrics.clustering import (
    CalinskiHarabaszScore,
    DaviesBouldinScore,
)

from sklearn.metrics import (
    precision_recall_curve,
)
from metrics.metrics import evaluate, combine_all_evaluation_scores
from utils.common_config import get_feature_dimensions_backbone
from data.custom_dataset import ContrustiveDataset
from losses.losses import entropy
from termcolor import colored
from utils.utils import find_target
import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import warnings

warnings.filterwarnings(
    "ignore", 
    message="No positive class found in y_true"
)

def SilhouetteScore(feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """GPU-compatible Silhouette Score. O(N^2) memory due to pairwise distances."""
    n = feats.shape[0]
    labels = labels.long()
    dists = torch.cdist(feats, feats)  # (n, n)

    a = torch.zeros(n, device=feats.device, dtype=feats.dtype)
    b = torch.full((n,), float("inf"), device=feats.device, dtype=feats.dtype)
    arange = torch.arange(n, device=feats.device)
    unique_labels = torch.unique(labels)

    for i in range(n):
        same_mask = (labels == labels[i]) & (arange != i)
        if same_mask.any():
            a[i] = dists[i, same_mask].mean()
            for c in unique_labels:
                if c == labels[i]:
                    continue
                other_mask = labels == c
                if other_mask.any():
                    avg = dists[i, other_mask].mean()
                    if avg < b[i]:
                        b[i] = avg
        else:
            # Single-element cluster: silhouette is defined as 0
            a[i] = 0.0
            b[i] = 0.0

    # Guard against remaining infinities (e.g. only one cluster present)
    b = torch.where(torch.isinf(b), torch.zeros_like(b), b)

    denom = torch.maximum(a, b)
    s = torch.zeros_like(a)
    valid = denom > 0
    s[valid] = (b[valid] - a[valid]) / denom[valid]
    return s.mean()

class GradientMonitor:
    def __init__(
        self,
        model: nn.Module,
        logger,
        log_interval: int = 10,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 1e3,
        log_histograms: bool = True,
        step: int = 0,
        aggregate: bool = False,
    ):
        self.model = model
        self.logger = logger
        self.log_interval = log_interval
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.log_histograms = log_histograms
        self.step_count = step
        self.aggregate = aggregate
        self._gradient_sums = {}
        self._ratio_sums = {}
        self._total_norm_sum = 0.0
        self._aggregate_steps = 0

    @torch.no_grad()
    def step(self) -> Dict[str, float]:
        self.step_count += 1
        total_norm_sq = 0.0
        names: List[str] = []
        norms: List[float] = []
        metrics: Dict[str, float] = {}

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad_norm = param.grad.norm(2).item()
            param_norm = param.norm(2).item()
            total_norm_sq += grad_norm ** 2

            tag = name.replace(".", "/")
            metrics[f"grad_norm/{tag}"] = grad_norm
            ratio = grad_norm / (param_norm + 1e-12)
            metrics[f"update_ratio/{tag}"] = ratio

            if self.aggregate:
                self._gradient_sums[name] = self._gradient_sums.get(name, 0.0) + grad_norm
                self._ratio_sums[name] = self._ratio_sums.get(name, 0.0) + ratio
            else:
                self.logger.scalar_summary("grad_norm", tag, grad_norm, self.step_count)
                self.logger.scalar_summary("update_ratio", tag, ratio, self.step_count)

            if self.log_histograms and not self.aggregate:
                try:
                    self.logger.add_histogram("grad_values", tag, param.grad, self.step_count)
                except ValueError as e:
                    print(f"Histogram value of :{name} has param.grad:{param.grad}")
                    import traceback
                    traceback.print_exc()
            names.append(name)
            norms.append(grad_norm)

        total_norm = total_norm_sq ** 0.5
        metrics["grad_norm/total"] = total_norm


        if self.aggregate:
            self._total_norm_sum += total_norm
            self._aggregate_steps += 1
            if self.step_count % self.log_interval != 0:
                return metrics

            count = self._aggregate_steps
            names = sorted(self._gradient_sums)
            norms = []
            for name in names:
                tag = name.replace(".", "/")
                average_norm = self._gradient_sums[name] / count
                average_ratio = self._ratio_sums[name] / count
                self.logger.scalar_summary("grad_norm", tag, average_norm, self.step_count)
                self.logger.scalar_summary("update_ratio", tag, average_ratio, self.step_count)
                norms.append(average_norm)
            total_norm = self._total_norm_sum / count
            self.logger.scalar_summary("grad_norm", "total", total_norm, self.step_count)
            self._gradient_sums.clear()
            self._ratio_sums.clear()
            self._total_norm_sum = 0.0
            self._aggregate_steps = 0
        else:
            self.logger.scalar_summary("grad_norm", "total", total_norm, self.step_count)

            # Push a gradient-flow bar chart as an image every N steps.
            if self.step_count % self.log_interval == 0:
                fig = self._plot_gradient_flow(names, norms)
                self.logger.add_figure("gradient_flow", fig, self.step_count)
                plt.close(fig)

        # Optional warnings (omitted here for brevity; re-add if desired)
        return metrics

    def _plot_gradient_flow(self, names: List[str], norms: List[float]):
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.4), 5))

        norms_arr = np.array(norms, dtype=float)

        # --- Detect problematic states ---------------------------------
        zero_mask = norms_arr == 0.0
        nan_mask = ~np.isfinite(norms_arr)
        vanishing_mask = (~zero_mask) & (~nan_mask) & (norms_arr < self.vanishing_threshold)
        exploding_mask = (~zero_mask) & (~nan_mask) & (norms_arr > self.exploding_threshold)

        # --- Prepare values for log-scale drawing ----------------------
        floor = 1e-10
        plot_vals = norms_arr.copy()
        plot_vals[zero_mask | nan_mask] = floor

        # --- Color coding ----------------------------------------------
        colors = []
        for i in range(len(norms_arr)):
            if zero_mask[i] or nan_mask[i]:
                colors.append("crimson")      # dead / zero
            elif vanishing_mask[i]:
                colors.append("orange")
            elif exploding_mask[i]:
                colors.append("green")
            else:
                colors.append("steelblue")    # healthy

        bars = ax.bar(
            range(len(names)),
            plot_vals,
            color=colors,
            edgecolor="black",
            linewidth=0.3,
        )

        # --- Annotate bars that are exactly zero -----------------------
        for bar, is_zero, is_nan in zip(bars, zero_mask, nan_mask):
            if is_zero or is_nan:
                label = "0" if is_zero else "NaN"
                ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    color="crimson",
                    fontweight="bold",
                )

        # --- Axis setup ------------------------------------------------
        if np.max(plot_vals) <= floor:
            # Every single gradient is zero / non-finite
            ax.set_yscale("linear")
            ax.set_ylim(-0.1, 1.0)
            ax.text(
                0.5,
                0.5,
                "All gradients are zero or non-finite",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                color="red",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )
        else:
            ax.set_yscale("log")
            ax.set_ylim(bottom=floor * 0.1)   # ensure zero-markers sit on the axis

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(
            [n.replace(".weight", "w").replace(".bias", "b") for n in names],
            rotation=45,
            ha="right",
            fontsize=6,
        )
        ax.set_ylabel("Loss Gradient Norm")
        ax.set_title("Gradient Flow")

        # --- Reference lines (only if visible) -------------------------
        y_min, y_max = ax.get_ylim()
        if self.vanishing_threshold >= y_min:
            ax.axhline(
                self.vanishing_threshold,
                color="orange",
                linestyle="--",
                linewidth=1,
            )
        if self.exploding_threshold <= y_max:
            ax.axhline(
                self.exploding_threshold,
                color="purple",
                linestyle="--",
                linewidth=1,
            )

        # --- Legend ----------------------------------------------------
        legend_elements = [
            Patch(facecolor="steelblue", label="healthy"),
            Patch(facecolor="orange", label="vanishing"),
            Patch(facecolor="green", label="exploding"),
            Patch(facecolor="crimson", label="zero / dead"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

        plt.tight_layout()
        return fig


@torch.no_grad()
def contrastive_evaluate(
    dataloader: torch.utils.data.DataLoader, model, output_metrics=True
):
    model.eval()
    device = next(model.parameters()).device

    all_feats = []
    all_meta = []
    for batch in dataloader:
        vertices = []
        labels = []
        ts_org = batch["ts_org"].to(device, non_blocking=True)
        target = find_target(batch["target"])

        b, w, h = ts_org.shape
        label = ["anchor" if t==0 else "anomaly" for t in target.tolist()]
        out = model(ts_org.reshape(b, h, w)).cpu()
        vertices.append(out)
        labels.append(label)

        if "ts_w_augment" in batch.keys():
            ts_w_augment = batch["ts_w_augment"].to(device, non_blocking=True)
            label = ["near neighbor" for _ in target.tolist()]
            out = model(ts_w_augment.reshape(b, h, w)).cpu()
            vertices.append(out)
            labels.append(label)

        if "ts_ss_augment" in batch.keys():
            ts_ss_augment = batch["ts_ss_augment"].to(device, non_blocking=True)
            label = ["far neighbor" for _ in target.tolist()]
            out = model(ts_ss_augment.reshape(b, h, w)).cpu()
            vertices.append(out)
            labels.append(label)

        all_feats.extend(vertices)
        all_meta.extend(labels)

    feats = torch.cat(all_feats, dim=0)
    metadata = [m for group in all_meta for m in group]

    # K-means clustering
    evaluation_metrics = {}
    if output_metrics:
        kmeans = MiniBatchKMeans(
            n_clusters=2,
            random_state=4,
            n_init="auto",
            batch_size=dataloader.batch_size,
        )
        cluster_labels = kmeans.fit_predict(feats.numpy())
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_)

        # Move data to the model device for GPU metric computation
        feats_dev = feats.to(device)
        labels_dev = torch.from_numpy(cluster_labels).to(device)

        # TorchMetrics objects (GPU-native)
        ch = CalinskiHarabaszScore().to(device)
        db = DaviesBouldinScore().to(device)

        try:
            s_score = SilhouetteScore(feats_dev, labels_dev).item()
            ch_score = ch(feats_dev, labels_dev).item()
            db_score = db(feats_dev, labels_dev).item()
        except ValueError as e:
            s_score = 0
            ch_score = 0
            db_score = 0
            print(colored(f"cluster_labels: {cluster_labels}", "red"))
            print(colored(f"{e}", "red"))

        # Add centroids to the graph
        feats = torch.cat([feats, cluster_centers], dim=0)
        metadata = metadata + ["anchor/near centroid", "far neighbor centroid"]
        evaluation_metrics = {
            "Silhouette Score": s_score,
            "Calinski-Harabasz Score": ch_score,
            "Davies-Bouldin Score": db_score,
        }
    return feats, metadata, evaluation_metrics


@torch.no_grad()
def get_predictions(p, dataloader, model, return_features=False, is_training=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    probs = []
    targets = []
    inputs = []
    start_idxs = []
    end_idxs = []

    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim))

    if isinstance(dataloader.dataset, ContrustiveDataset):  # Also return the neighbors
        key_ = "anchor"
        include_neighbors = True
        nneighbors = []
        fneighbors = []

    else:
        key_ = "ts_org"
        include_neighbors = False

    ptr = 0
    for batch in dataloader:
        ts = batch[key_]
        meta = batch["meta"]
        # ts = torch.unsqueeze(ts, dim=1)
        if ts.ndim == 3:
            bs, w, h = ts.shape
        else:
            bs, w = ts.shape
            h = 1

        target = batch["target"]
        
        targets.append(target)
        inputs.append(ts.cpu())

        res = model(ts.reshape(bs, h, w).to(device), forward_pass="return_all")
        output = res["output"]
        if return_features:
            features[ptr : ptr + bs] = res["features"]
            ptr += bs
        predictions.append(torch.argmax(output, dim=1))
        probs.append(F.softmax(output, dim=1) if output.size(1) > 1 else F.sigmoid(output))
        start_idxs.append(meta["start_idx"])
        end_idxs.append(meta["end_idx"])

        if include_neighbors:
            nneighbors.append(batch["possible_nneighbors"])
            fneighbors.append(batch["possible_fneighbors"])

    predictions = torch.cat(predictions, dim=0).cpu()
    probs = torch.cat(probs, dim=0).cpu()
    targets = torch.cat(targets, dim=0)
    inputs = torch.cat(inputs, dim=0)
    start_idxs = torch.cat(start_idxs, dim=0)
    end_idxs = torch.cat(end_idxs, dim=0) 
    out = {
            "predictions": predictions,
            "probabilities": probs,
            "targets": targets,
            "inputs": inputs,
            "start_idxs": start_idxs,
            "end_idxs": end_idxs,
        }
    
    if include_neighbors:
        nneighbors = torch.cat(nneighbors, dim=0)
        fneighbors = torch.cat(fneighbors, dim=0)
        out["neighbors"] = nneighbors
        out["fneighbors"] = fneighbors

    if return_features:
        feat_np = features.numpy()  # save features in csv
        fhdr = [str(x) for x in range(feat_np.shape[1])] + ["Class"]

        final_targets = find_target(targets)
        feat_np = np.hstack((feat_np, final_targets[np.newaxis].T))

        feat_df = pd.DataFrame(feat_np, columns=fhdr)

        prob_np = np.array(out["probabilities"])
        phdr = [str(x) for x in range(prob_np.shape[1])] + ["Class"]

        prob_np = np.hstack((prob_np, final_targets[np.newaxis].T))
        prob_df = pd.DataFrame(prob_np, columns=phdr)

        if is_training:
            feat_df.to_csv(
                p["classification_trainfeatures"], index=False, header=True, sep=","
            )
            prob_df.to_csv(
                p["classification_trainprobs"], index=False, header=True, sep=","
            )
        else:
            feat_df.to_csv(
                p["classification_testfeatures"], index=False, header=True, sep=","
            )
            prob_df.to_csv(
                p["classification_testprobs"], index=False, header=True, sep=","
            )

        return out, features.cpu()

    else:
        return out


@torch.no_grad()
def classification_evaluate(predictions):
    # Evaluate model based on classification loss.

    # Neighbors and anchors
    probs = predictions["probabilities"]
    neighbors = predictions["neighbors"]
    fneighbors = predictions["fneighbors"]
    org_anchors = torch.arange(neighbors.size(0)).view(-1, 1).expand_as(neighbors)

    # Entropy loss
    entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

    # Consistency loss
    similarity = torch.matmul(probs, probs.t())
    neighbors = neighbors.contiguous().view(-1)
    anchors = org_anchors.contiguous().view(-1)
    similarity = similarity[anchors, neighbors]
    ones = torch.ones_like(similarity)
    consistency_loss = F.binary_cross_entropy(similarity, ones).item()

    similarity = torch.matmul(probs, probs.t())
    fneighbors = fneighbors.contiguous().view(-1)
    anchors = org_anchors.contiguous().view(-1)
    similarity = similarity[anchors, fneighbors]
    ones = torch.ones_like(similarity)
    inconsistency_loss = F.binary_cross_entropy(similarity, ones).item()

    # Total loss
    total_loss = 5 * entropy_loss + consistency_loss - 0 * inconsistency_loss

    output = {
        "entropy": entropy_loss,
        "consistency": consistency_loss,
        "inconsistency": inconsistency_loss,
        "total_loss": total_loss,
    }
    return {"classification": output}

@torch.no_grad()
def pr_evaluate_timeseries(
    logger,
    all_predictions,
    train_best_threshold,
    majority_label,
    gt,
    inputs,
    tag,
    epoch=-1,
    ch=None,
    make_figures=True,
):

    probs = all_predictions["probabilities"]
    predictions = all_predictions["predictions"]
    start = all_predictions["start_idxs"].numpy().astype(int)
    end = all_predictions["end_idxs"].numpy().astype(int)

    # Classification metrics
    targets = (gt != 0).astype(int)
    predictions = (predictions != majority_label).numpy().astype(int)
    predictions = np.repeat(predictions[:, np.newaxis], (end[0]-start[0]), axis=-1)
    cls_score, best_detections_thresholds = evaluate(logger, epoch, predictions, inputs, targets, start, end, tag=f"{tag}_cls_prediction_", ch=ch, threshold=None, det_threshold=None, pre_classify=False, make_figures=make_figures)

    # Anomaly score metrics
    scores = 1 - np.array(probs)[:, majority_label]
    scores = np.repeat(scores[:, np.newaxis], (end[0]-start[0]), axis=-1)
    # Find best threshold based on F1 score
    score_best, threshold_best = evaluate(logger, epoch, scores, inputs, targets, start, end, tag=f"{tag}_anomaly_best_", ch=ch, threshold=None, det_threshold=1, pre_classify=False, make_figures=make_figures)

    # Anomaly score metrics based on train best threshold
    score_train_best, _ = evaluate(logger, epoch, scores, inputs, targets, start, end, tag=f"{tag}_anomaly_train_best_", ch=ch, threshold=train_best_threshold, det_threshold=1, pre_classify=False, make_figures=make_figures, make_extras=False)
    return cls_score, score_best, score_train_best, threshold_best, best_detections_thresholds

@torch.no_grad()
def pr_evaluate(
    all_predictions,
    majority_label=0,
    train=False,
    train_best_threshold=None,
):

    targets = find_target(all_predictions["targets"])
    predictions = all_predictions["predictions"]
    probs = all_predictions["probabilities"]

    # Classification metrics
    cls_targets = np.where((targets == 4), 1, 0) if train else np.where((targets == 0), 0, 1)
    anomalies = np.where((predictions == majority_label), 0, 1)

    cls_score = combine_all_evaluation_scores(anomalies, cls_targets)

    # Anomaly score metrics
    scores = 1 - np.array(probs)[:, majority_label]
    labels = cls_targets.astype(int)
    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    f1_score = 2 * precision * recall / (precision + recall)
    if np.isnan(f1_score).any():
        f1_score = np.nan_to_num(f1_score)

    best_f1_index = np.argmax(f1_score)
    best_threshold = thresholds[best_f1_index]
    anomalies = np.where((scores >= best_threshold), 1, 0)
    score_best = combine_all_evaluation_scores(anomalies, labels)

    # Anomaly score metrics based on train best threshold
    score_train_best = {}
    if not train and (train_best_threshold is not None):# and probs.size(1) == 1:
            anomalies = np.where((scores >= train_best_threshold), 1, 0)
            score_train_best = combine_all_evaluation_scores(anomalies, labels)
    return cls_score, score_best, score_train_best, best_threshold
