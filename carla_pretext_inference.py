import argparse
import csv
import os
import random
from collections.abc import Mapping

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC, OneClassSVM

from metrics.metrics import combine_all_evaluation_scores
from utils.common_config import (
    get_model,
    get_train_dataset,
    get_train_transformations,
    get_val_dataset,
    get_val_dataloader,
    get_val_transformations,
    inject_sub_anomaly,
)
from utils.config import create_config
from utils.utils import Logger, clean_checkpoint, find_target, mkdir_if_missing


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(4)


def get_checkpoint_path(p, checkpoint):
    # "model" -> best model saved during pretext training
    # "last"  -> last epoch checkpoint
    # anything else is treated as a metric name (e.g. loss, clear_loss,
    # train_calinski, train_davies, train_silhouette, eval_calinski,
    # eval_davies, eval_silhouette, best)
    if checkpoint == "model":
        return p["pretext_model"]
    if checkpoint == "last":
        return p["pretext_checkpoint"]
    return f"{p['pretext_checkpoint'][:-4]}_{checkpoint}.pth.tar"


def load_pretext_model(p, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise ValueError(
            "Path with pre-trained weights does not exist {}".format(checkpoint_path)
        )
    model = get_model(p)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = (
        checkpoint["model"]
        if (isinstance(checkpoint, Mapping) and "model" in checkpoint)
        else checkpoint
    )
    state = clean_checkpoint(state)
    model.load_state_dict(state)
    return model.to(device)


@torch.no_grad()
def extract_features(dataloader, model, keys=("ts_org",)):
    # Forward every input through the pretext model and collect the
    # contrastive embeddings. Mirrors `contrastive_evaluate` in
    # utils.evaluate_utils.
    model.eval()
    device = next(model.parameters()).device

    feats = {key: [] for key in keys}
    all_targets = []
    for batch in dataloader:
        b, w, h = batch["ts_org"].shape
        for key in keys:
            ts = batch[key].to(device, non_blocking=True)
            out = model(ts.reshape(b, h, w)).cpu()
            feats[key].append(out)
        all_targets.append(find_target(batch["target"]))

    feats = {key: torch.cat(val, dim=0).numpy() for key, val in feats.items()}
    targets = np.concatenate(all_targets).astype(int)
    return feats, targets


def fit_detector(method, train_feats, train_targets, args):
    # Fit the anomaly detector on the embeddings of the training inputs.
    # Returns the fitted detector and an info string.
    anchors = train_feats["ts_org"]
    normal_feats = anchors[train_targets == 0]
    if len(normal_feats) == 0:
        normal_feats = anchors

    if method == "svm":
        # Learn the boundary of the cluster of normal training embeddings.
        # Samples inside the boundary are normal, outside of it anomalous.
        detector = OneClassSVM(kernel=args.kernel, gamma=args.gamma, nu=args.nu)
        detector.fit(normal_feats)
        info = (
            f"OneClassSVM (kernel={args.kernel}, gamma={args.gamma}, nu={args.nu}) "
            f"on {len(normal_feats)} normal train embeddings"
        )

    elif method == "svc":
        # Supervised boundary between the normal cluster (anchors) and the
        # anomalous cluster (sub-anomaly augmentations, plus real anomalous
        # training windows when the training split provides them).
        anomaly_feats = [train_feats["ts_ss_augment"]]
        if (train_targets != 0).any():
            anomaly_feats.append(anchors[train_targets != 0])
        anomaly_feats = np.concatenate(anomaly_feats, axis=0)

        X = np.concatenate([normal_feats, anomaly_feats], axis=0)
        y = np.concatenate(
            [np.zeros(len(normal_feats)), np.ones(len(anomaly_feats))]
        ).astype(int)
        detector = SVC(
            kernel=args.kernel, gamma=args.gamma, C=args.C, probability=True, random_state=4
        )
        detector.fit(X, y)
        info = (
            f"SVC (kernel={args.kernel}, gamma={args.gamma}, C={args.C}) "
            f"on {len(normal_feats)} normal / {len(anomaly_feats)} anomalous train embeddings"
        )

    elif method == "iforest":
        # Isolation forest fitted on the cluster of normal training embeddings.
        detector = IsolationForest(
            n_estimators=args.n_estimators,
            contamination=args.contamination,
            random_state=4,
        )
        detector.fit(normal_feats)
        info = (
            f"IsolationForest (n_estimators={args.n_estimators}, "
            f"contamination={args.contamination}) on {len(normal_feats)} normal train embeddings"
        )

    elif method == "kmeans":
        # Cluster the normal training embeddings (same KMeans as
        # `contrastive_evaluate`). The anomaly score is the distance to the
        # nearest centroid of the normal cluster; the decision boundary is a
        # quantile of the distances on the training inputs.
        detector = MiniBatchKMeans(
            n_clusters=args.n_clusters,
            random_state=4,
            n_init="auto",
        )
        detector.fit(normal_feats)
        train_scores = detector.transform(normal_feats).min(axis=1)
        detector.threshold_ = float(np.quantile(train_scores, args.quantile))
        info = (
            f"MiniBatchKMeans (n_clusters={args.n_clusters}, quantile={args.quantile}, "
            f"threshold={detector.threshold_:.6f}) on {len(normal_feats)} normal train embeddings"
        )

    elif method == "kmeans2":
        # Two clusters over every training embedding: anchors together with
        # their near and far neighbors. Anchors and near neighbors belong to
        # normality, so the cluster holding the most training data is the
        # normal cluster; the other one is the anomalous cluster.
        all_feats = np.concatenate(
            [train_feats["ts_org"], train_feats["ts_w_augment"], train_feats["ts_ss_augment"]],
            axis=0,
        )
        detector = MiniBatchKMeans(n_clusters=2, random_state=4, n_init="auto")
        train_assignment = detector.fit_predict(all_feats)
        counts = np.bincount(train_assignment)
        detector.normal_label_ = int(np.argmax(counts))
        detector.anomaly_label_ = int(np.argmin(counts))
        info = (
            f"MiniBatchKMeans (n_clusters=2) on {len(all_feats)} train embeddings "
            f"(anchors + near + far neighbors); normal cluster: "
            f"{detector.normal_label_} ({counts[detector.normal_label_]} samples), "
            f"anomalous cluster: {detector.anomaly_label_} ({counts[detector.anomaly_label_]} samples)"
        )

    else:
        raise ValueError("Invalid method {}".format(method))

    return detector, info


def detector_predict(detector, method, feats):
    # Returns (scores, anomalies). Higher score means further outside the
    # normal cluster (more anomalous). anomalies are in {0, 1}.
    if method == "svm":
        # decision_function > 0 -> inside the normal cluster (inlier)
        # decision_function <= 0 -> outside of it (anomaly)
        decision = detector.decision_function(feats)
        scores = -decision
        anomalies = (decision <= 0).astype(int)

    elif method == "svc":
        normal_idx = list(detector.classes_).index(0)
        scores = 1 - detector.predict_proba(feats)[:, normal_idx]
        anomalies = (detector.predict(feats) != 0).astype(int)

    elif method == "iforest":
        scores = -detector.score_samples(feats)
        anomalies = (detector.predict(feats) == -1).astype(int)

    elif method == "kmeans":
        # Distance to the nearest centroid of the normal cluster
        scores = detector.transform(feats).min(axis=1)
        anomalies = (scores > detector.threshold_).astype(int)

    elif method == "kmeans2":
        # Normal if the nearest centroid is the normal cluster, anomalous if
        # it belongs to the anomalous cluster. Score: distance to the normal
        # centroid minus distance to the anomalous one (higher = anomalous).
        dists = detector.transform(feats)
        scores = dists[:, detector.normal_label_] - dists[:, detector.anomaly_label_]
        anomalies = (detector.predict(feats) != detector.normal_label_).astype(int)

    else:
        raise ValueError("Invalid method {}".format(method))

    return scores, anomalies


def boundary_threshold(detector, method):
    # Anomaly score value that corresponds to the detector's own decision
    # boundary. Used as the train threshold when the training split is
    # anomaly-free and no F1-optimal threshold can be derived from it.
    if method == "svm":
        return 0.0
    if method == "svc":
        return 0.5
    if method == "iforest":
        return float(-detector.offset_)
    if method == "kmeans":
        return detector.threshold_
    if method == "kmeans2":
        return 0.0  # equidistant from both centroids
    raise ValueError("Invalid method {}".format(method))


def best_f1_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_score = 2 * precision * recall / (precision + recall)
    f1_score = np.nan_to_num(f1_score)
    best_f1_index = np.argmax(f1_score)
    best_threshold = thresholds[min(best_f1_index, len(thresholds) - 1)]
    return best_threshold


def pr_evaluate_detector(scores, detector_anomalies, targets, train_best_threshold=None):
    # Mirror of `pr_evaluate` in utils.evaluate_utils for detector anomaly scores.
    labels = (targets != 0).astype(int)

    # Classification metrics (raw detector decision boundary)
    cls_score = combine_all_evaluation_scores(detector_anomalies, labels)

    # Anomaly score metrics with the best F1 threshold
    best_threshold = best_f1_threshold(labels, scores)
    anomalies = (scores >= best_threshold).astype(int)
    score_best = combine_all_evaluation_scores(anomalies, labels)

    # Anomaly score metrics based on train best threshold
    score_train_best = {}
    if train_best_threshold is not None:
        anomalies = (scores >= train_best_threshold).astype(int)
        score_train_best = combine_all_evaluation_scores(anomalies, labels)
    return cls_score, score_best, score_train_best, best_threshold


def save_metrics_csv(path, metrics):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, metrics.keys())
        w.writeheader()
        w.writerow(metrics)


def main(args, update_dictionary={}):
    p = create_config(args.config_env, args.config_exp, args.fname, args.version, update_dictionary=update_dictionary)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and p.get("device", "cuda")) else "cpu")

    # Parse detector hyperparameters that can be either str or float
    try:
        args.gamma = float(args.gamma)
    except ValueError:
        pass  # "scale" or "auto"
    try:
        args.contamination = float(args.contamination)
    except ValueError:
        pass  # "auto"

    inference_dir = os.path.join(
        p["pretext_dir"], f"{args.method}_inference{('_' + args.tag) if args.tag else ''}"
    )
    mkdir_if_missing(inference_dir)
    logger = Logger(
        p["version"], verbose=2, file_path=inference_dir, use_tensorboard=False, file_name=f"{args.method}_inference_log"
    )
    logger.log("CARLA Pretext Inference --> {} on contrastive embeddings".format(args.method))
    logger.log_hyperparams(p)

    # Model initialised from the pretext checkpoint
    checkpoint_path = get_checkpoint_path(p, args.checkpoint)
    logger.log("Load pretext model from {}".format(checkpoint_path))
    model = load_pretext_model(p, checkpoint_path, device)

    # Data - identical pipeline to carla_pretext.py
    train_transforms = get_train_transformations(p)
    sanomaly = inject_sub_anomaly(p)
    val_transforms = get_val_transformations(p)

    train_dataset = get_train_dataset(
        p, train_transforms, sanomaly, to_augmented_dataset=True
    )
    val_dataset = get_val_dataset(
        p, val_transforms, sanomaly, False, train_dataset.mean, train_dataset.std
    )

    # No shuffling and no dropped last batch so every training input is evaluated
    train_dataloader = get_val_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)

    logger.log(
        "Dataset contains {}/{} train/val samples".format(
            len(train_dataset), len(val_dataset)
        )
    )

    # Embeddings of every training/test input from the pretext model.
    # The supervised SVC also needs the sub-anomaly augmentations of the
    # training inputs to form the anomalous cluster.
    if args.method == "svc":
        train_keys = ("ts_org", "ts_ss_augment")
    elif args.method == "kmeans2":
        train_keys = ("ts_org", "ts_w_augment", "ts_ss_augment")
    else:
        train_keys = ("ts_org",)
    train_feats, train_targets = extract_features(train_dataloader, model, keys=train_keys)
    test_feats, test_targets = extract_features(val_dataloader, model)
    logger.log(
        "Extracted embeddings: train {} / test {}".format(
            train_feats["ts_org"].shape, test_feats["ts_org"].shape
        )
    )

    # Detector on the cluster(s) of training embeddings
    detector, info = fit_detector(args.method, train_feats, train_targets, args)
    logger.log("Fitted {}".format(info))
    joblib.dump(detector, os.path.join(inference_dir, f"{args.method}_model.joblib"))

    # Anomaly scores (higher = further outside the normal cluster)
    train_scores, train_anomalies = detector_predict(detector, args.method, train_feats["ts_org"])
    test_scores, test_anomalies = detector_predict(detector, args.method, test_feats["ts_org"])

    # Best anomaly threshold derived from the training inputs. The training
    # split can be anomaly-free (single class); in that case the detector's
    # own decision boundary is the train threshold.
    if len(np.unique(train_targets)) > 1:
        train_best_threshold = best_f1_threshold(
            (train_targets != 0).astype(int), train_scores
        ).item()
    else:
        train_best_threshold = boundary_threshold(detector, args.method)

    # Evaluation on the test split
    cls_score, score_best, score_train_best, best_threshold = pr_evaluate_detector(
        test_scores, test_anomalies, test_targets, train_best_threshold
    )

    # Save embeddings, scores and metrics
    np.save(os.path.join(inference_dir, "train_features.npy"), train_feats["ts_org"])
    np.save(os.path.join(inference_dir, "train_targets.npy"), train_targets)
    np.save(os.path.join(inference_dir, "train_scores.npy"), train_scores)
    np.save(os.path.join(inference_dir, "test_features.npy"), test_feats["ts_org"])
    np.save(os.path.join(inference_dir, "test_targets.npy"), test_targets)
    np.save(os.path.join(inference_dir, "test_scores.npy"), test_scores)

    save_metrics_csv(os.path.join(inference_dir, "eval_test_cls.csv"), cls_score)
    save_metrics_csv(os.path.join(inference_dir, "eval_test_best.csv"), score_best)
    if score_train_best:
        save_metrics_csv(
            os.path.join(inference_dir, "eval_test_train_th.csv"), score_train_best
        )

    # Evaluation on the training inputs (only meaningful if it has both classes)
    if len(np.unique(train_targets)) > 1:
        train_cls_score, train_best, _, _ = pr_evaluate_detector(
            train_scores, train_anomalies, train_targets
        )
        save_metrics_csv(
            os.path.join(inference_dir, "eval_train_cls.csv"), train_cls_score
        )
        save_metrics_csv(
            os.path.join(inference_dir, "eval_train_best.csv"), train_best
        )

    report_str = (
        f"\nPretext {args.method} Inference\n"
        f"Checkpoint: {checkpoint_path}\n"
        f"{info}\n"
        f"Train anomalies (boundary): {int(train_anomalies.sum())}/{len(train_anomalies)}\n"
        f"Test anomalies (boundary): {int(test_anomalies.sum())}/{len(test_anomalies)}\n"
        f"Decision Boundary Classification -->\n"
        f"{''.join(f'{key}:{value}\n' for key, value in cls_score.items())}"
        f"Anomalies Best F1 --> Best Threshold: {best_threshold}\n"
        f"{''.join(f'{key}:{value}\n' for key, value in score_best.items())}"
        f"Anomalies Train Best F1 --> Threshold: {train_best_threshold}\n"
        f"{''.join(f'{key}:{value}\n' for key, value in score_train_best.items())}"
    )
    logger.log(report_str)
    logger.log("Results saved in {}".format(inference_dir))
    logger.finalize()


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description="pretext inference")
    parser.add_argument("--config_env", help="Config file for the environment", default="configs/env.yml")
    parser.add_argument("--config_exp", help="Config file for the experiment", default="configs/pretext/new_loss/smd/normalization-strategy/dynamic_margin_by_neg_distance-dynamic_loss_guidance-clamp_only_negative_loss-dynamic_weight_loss.yml")
    parser.add_argument("--fname", help="Config the file name of Dataset", default="machine-1-1.txt")
    parser.add_argument("--version", help="Experiment version", type=str, default="best_models/batch/dynamic_margin_by_neg_distance-dynamic_loss_guidance-clamp_only_negative_loss-dynamic_weight_loss")
    parser.add_argument(
        "--method",
        help="Anomaly detector fitted on the training embeddings: 'svm' "
        "(OneClassSVM on the normal cluster), 'svc' (supervised SVM separating "
        "the normal cluster from the sub-anomaly augmented cluster), "
        "'iforest' (IsolationForest on the normal cluster), 'kmeans' "
        "(KMeans on the normal cluster, distance to the nearest centroid) or "
        "'kmeans2' (2 KMeans clusters over anchors + near + far neighbors, the "
        "biggest cluster is the normal one)",
        choices=["svm", "svc", "iforest", "kmeans", "kmeans2"],
        default="svm",
    )
    parser.add_argument(
        "--checkpoint",
        help="Pretext weights to load: 'model' (default), 'last' or a metric name "
        "(loss, clear_loss, train_calinski, train_davies, train_silhouette, "
        "eval_calinski, eval_davies, eval_silhouette, best)",
        type=str,
        default="clear_loss",
    )
    # SVM / SVC hyperparameters
    parser.add_argument(
        "--nu",
        help="OneClassSVM nu (method=svm): upper bound on the fraction of "
        "training errors and lower bound of the fraction of support vectors",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--kernel", help="SVM/SVC kernel", type=str, default="rbf"
    )
    parser.add_argument(
        "--gamma",
        help="SVM/SVC gamma ('scale', 'auto' or a float)",
        type=str,
        default="scale",
    )
    parser.add_argument(
        "--C", help="SVC regularization parameter (method=svc)", type=float, default=1.0
    )
    # IsolationForest hyperparameters
    parser.add_argument(
        "--n_estimators",
        help="IsolationForest number of trees (method=iforest)",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--contamination",
        help="IsolationForest contamination ('auto' or a float, method=iforest)",
        type=str,
        default="auto",
    )
    # KMeans hyperparameters
    parser.add_argument(
        "--n_clusters",
        help="KMeans number of clusters of the normal embeddings (method=kmeans)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--quantile",
        help="Quantile of the train distances to the nearest centroid used as "
        "the KMeans decision boundary (method=kmeans)",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--tag",
        help="Optional tag appended to the inference output directory",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    main(args)
