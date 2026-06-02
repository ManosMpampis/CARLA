import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from sklearn.metrics import (
    precision_recall_curve,
    # confusion_matrix,
    # multilabel_confusion_matrix,
)
from metrics.metrics import evaluate, combine_all_evaluation_scores
from utils.common_config import get_feature_dimensions_backbone
from data.custom_dataset import NeighborsDataset
from losses.losses import entropy
from termcolor import colored

import warnings

warnings.filterwarnings(
    "ignore", 
    message="No positive class found in y_true"
)


@torch.no_grad()
def contrastive_evaluate(
    dataloader: torch.utils.data.DataLoader, model, output_metrics=True
):
    model.eval()
    device = next(model.parameters()).device

    all_feats = []
    all_meta = []
    for batch in dataloader:
        ts_org = batch["ts_org"].to(device, non_blocking=True)
        ts_w_augment = batch["ts_w_augment"].to(device, non_blocking=True)
        ts_ss_augment = batch["ts_ss_augment"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        b, w, h = ts_org.shape
        target_str = [str(l) for l in target.tolist()]
        vertices_org = model(ts_org.view(b, h, w)).cpu()
       
        target_w_str = [str(l * 2) for l in torch.ones_like(target).tolist()]
        vertices_w = model(ts_w_augment.view(b, h, w)).cpu()
        
        target_ss = torch.ones_like(target)
        target_ss_str = [str(l) for l in target_ss.tolist()]
        vertices_ss = model(ts_ss_augment.view(b, h, w)).cpu()

        all_feats.extend([vertices_org, vertices_w, vertices_ss])
        all_meta.extend([target_str, target_w_str, target_ss_str])

    feats = torch.cat(all_feats, dim=0)
    metadata = [m for group in all_meta for m in group]

    # K-means clustering
    evaluation_metrics = {}
    if output_metrics:
        kmeans = MiniBatchKMeans(
            n_clusters=3,
            random_state=4,
            n_init="auto",
            batch_size=dataloader.batch_size,
        )
        cluster_labels = kmeans.fit_predict(feats.numpy())
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_)

        try:
            # Calculate Silhouette Score
            s_score = metrics.silhouette_score(
                feats.numpy(), cluster_labels, metric="euclidean"
            )
            # Calculate Calinski-Harabasz Index
            ch_score = metrics.calinski_harabasz_score(feats.numpy(), cluster_labels)
            # Calculate Davies-Bouldin Index
            db_score = metrics.davies_bouldin_score(feats.numpy(), cluster_labels)
        except ValueError as e:
            s_score = 0
            ch_score = 0
            db_score = 0
            print(colored(f"cluster_labels: {cluster_labels}", "red"))
            print(colored(f"{e}", "red"))

        # Add centroids to the graph
        feats = torch.cat([feats, cluster_centers], dim=0)
        metadata = metadata + ["centroid 0", "centroid 1", "centroid 2"]
        evaluation_metrics = {
            "Silhouette Score": s_score,
            "Calinski-Harabasz Score": ch_score,
            "Davies-Bouldin Score": db_score,
        }
    return feats, metadata, evaluation_metrics


@torch.no_grad()
def get_predictions(p, dataloader, model, return_features=False, is_training=False):
    # Make predictions on a dataset with neighbors
    global features, nneighbors, fneighbors
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
        features = torch.zeros((len(dataloader.sampler), ft_dim))  # .cuda()

    if isinstance(dataloader.dataset, NeighborsDataset):  # Also return the neighbors
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

        
        if isinstance(ts, np.ndarray):
            ts = torch.from_numpy(ts).float()
            targets.append(torch.from_numpy(batch["target"]))
        else:
            targets.append(batch["target"])
        inputs.append(ts.cpu())
        res = model(ts.view(bs, h, w).to(device), forward_pass="return_all")
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
        # feat_np = np.hstack((feat_np, np.array(targets)[np.newaxis].T)) CUDA
        feat_np = np.hstack((feat_np, np.array(targets.cpu().numpy())[np.newaxis].T))

        feat_df = pd.DataFrame(feat_np, columns=fhdr)

        prob_np = np.array(out["probabilities"])
        phdr = [str(x) for x in range(prob_np.shape[1])] + ["Class"]
        # prob_np = np.hstack((prob_np, np.array(targets)[np.newaxis].T))
        prob_np = np.hstack((prob_np, np.array(targets.cpu().numpy())[np.newaxis].T))
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
        # tmp = np.array(out[0]['probabilities'])
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
    ch=None
):

    probs = all_predictions["probabilities"]
    predictions = all_predictions["predictions"]
    start = all_predictions["start_idxs"]
    end = all_predictions["end_idxs"]

    # Classification metrics
    targets = (gt == 0).astype(int)
    predictions = (predictions == majority_label).astype(int)
    cls_score, best_detections_thresholds = evaluate(logger, epoch, predictions, inputs, targets, start, end, tag=f"{tag}cls_prediction", ch=ch, threshold=0.5, det_threshold=1, pre_classify=False, make_figures=True)

    # Anomaly score metrics
    scores = 1 - np.array(probs)[:, majority_label]
    # Find best threshold based on F1 score
    score_best, threshold_best = evaluate(logger, epoch, scores, inputs, targets, start, end, tag=f"{tag}anomaly_best", ch=ch, threshold=None, det_threshold=1, pre_classify=False, make_figures=True)

    # Anomaly score metrics based on train best threshold
    score_train_best, _ = evaluate(logger, epoch, scores, inputs, targets, start, end, tag=f"{tag}anomaly_train_best", ch=ch, threshold=train_best_threshold, det_threshold=1, pre_classify=False, make_figures=True)
    return cls_score, score_best, score_train_best, threshold_best, best_detections_thresholds

@torch.no_grad()
def pr_evaluate(
    all_predictions,
    majority_label=0,
    train=False,
    train_best_threshold=None,
):

    targets = all_predictions["targets"]
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
    if not train and (train_best_threshold is not None) and probs.size(1) == 1:
            anomalies = np.where((scores >= train_best_threshold), 1, 0)
            score_train_best = combine_all_evaluation_scores(anomalies, labels)
    return cls_score, score_best, score_train_best

# %% 
# @torch.no_grad()
# def pr_evaluate(
#     all_predictions,
#     majority_label=0,
#     train=False,
#     train_best_threshold=None,
# ):

#     targets = all_predictions["targets"].cpu()  # .cuda()
#     predictions = all_predictions["predictions"].cpu()  # .cuda()
#     probs = all_predictions["probabilities"].cpu()  # .cuda()

#     cls_targets = np.where((targets == 4), 1, 0) if train else np.where((targets == 0), 0, 1)
#     scores = np.array(probs)

#     tns, tps, fps, fns, thr = confusion_matrix_at_thresholds(cls_targets, probs)
#     pre = tps / (tps + fps) if (tps + fps) != 0 else 0
#     recall = tps / (tps + fns)
    
#     labels = cls_targets.tolist() #np.array(targets.cpu().numpy()).tolist()
#     try:
#         f_1 = 2 * pre * recall / (pre + recall)
#         if np.isnan(f1_score).any():
#             f1_score = np.nan_to_num(f1_score)
#             # print("f1: Nan --> 0")
#     except ZeroDivisionError:
#         f1_score = [0.0]
#         print("f1: 0 --> 0")

#     best_f1_index = np.argmax(f1_score)

#     rep_f1 = f1_score[best_f1_index]

#     best_threshold = thr[best_f1_index]
#     best_precision = pre[best_f1_index]
#     best_recall = recall[best_f1_index]

#     best_tn = tns[best_f1_index]
#     best_tp = tps[best_f1_index]
#     best_fn = fns[best_f1_index]
#     best_fp = fps[best_f1_index]


#     anomalies = [1 if s >= best_threshold else 0 for s in scores]
#     out = {
#         "best_tp": best_tp,
#         "best_tn": best_tn,
#         "best_fp": best_fp,
#         "best_fn": best_fn,
#         "best_th": best_threshold,
#         "best_pre": best_precision,
#         "best_rec": best_recall,
#         "best_f1": rep_f1,
#     }
#     if not train and (train_best_threshold is not None) and probs.size(1) == 1:
#             anomalies = [1 if s >= train_best_threshold else 0 for s in scores]
#             tn_train, fp_train, fn_train, tp_train = confusion_matrix(labels, anomalies).ravel()
#             train_precision = tp_train / (tp_train + fp_train) if (tp_train + fp_train) != 0 else 0
#             train_recall = tp_train / (tp_train + fn_train)
#             train_f1 = 2 * train_precision * train_recall / (train_precision + train_recall) if (train_precision + train_recall) != 0 else 0
#             out["best_train_tp"] = tn_train
#             out["best_train_fp"] = fp_train
#             out["best_train_tn"] = tp_train
#             out["best_train_fn"] = fn_train
#             out["best_train_th"] = train_best_threshold
#             out["best_train_pre"] = train_precision
#             out["best_train_rec"] = train_recall
#             out["best_train_f1"] = train_f1
#     return out
# %% 
def replace_majority_label(flat_preds, majority_label):
    # unique_labels = torch.unique(flat_preds)
    new_pred = torch.where(flat_preds == majority_label, 0, 1)
    return new_pred

