import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn import metrics
from torchmetrics.functional.classification import confusion_matrix
from torchmetrics.functional import precision_recall_curve

from utils.common_config import get_feature_dimensions_backbone
from utils.utils import EmptyLogger
from data.custom_dataset import NeighborsDataset
from losses.losses import entropy


@torch.no_grad()
def contrastive_evaluate(dataloader, model):
    model.eval()
    device = next(model.parameters()).device
    
    # K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=4, n_init="auto")

    all_feats = []
    all_meta = []
    for batch in dataloader:
        ts_org = batch['ts_org'].to(device, non_blocking=True)
        b, w, h = ts_org.shape
        target = batch['target'].to(device, non_blocking=True)
        target_str = [str(l) for l in target.tolist()]

        vertices_org = model(ts_org.view(b, h, w)).cpu()

        ts_w_augment = batch['ts_w_augment'].to(device, non_blocking=True)
        target_w = target.copy()
        target_w_str = [str(l*2) for l in torch.ones_like(target).tolist()]
        vertices_w = model(ts_w_augment.view(b, h, w)).cpu()

        ts_ss_augment = batch['ts_ss_augment'].to(device, non_blocking=True)
        target_ss = torch.ones_like(target)
        target_ss_str = [str(l) for l in target_ss.tolist()]
        vertices_ss = model(ts_ss_augment.view(b, h, w)).cpu()

        all_feats.extend([vertices_org, vertices_w, vertices_ss])
        all_meta.extend([target_str, target_w_str, target_ss_str])

    feats = torch.cat(all_feats, dim=0)
    metadata = [m for group in all_meta for m in group]
    
    cluster_labels = kmeans.fit_predict(feats.numpy())

    # Calculate Silhouette Score
    s_score = metrics.silhouette_score(feats.numpy(), cluster_labels, metric='euclidean')
    # Calculate Calinski-Harabasz Index
    ch_score = metrics.calinski_harabasz_score(feats.numpy(), cluster_labels)
    # Calculate Davies-Bouldin Index
    db_score = metrics.davies_bouldin_score(feats.numpy(), cluster_labels)
    evaluation_metrics = {"Silhouette Score": s_score, "Calinski-Harabasz Score": ch_score, "Davies-Bouldin Score": db_score}
    return feats, metadata, evaluation_metrics


@torch.no_grad()
def get_predictions(p, dataloader, model, return_features=False, is_training=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    device = next(model.parameters()).device

    predictions = []
    probs = []
    targets = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim))

    if isinstance(dataloader.dataset, NeighborsDataset): # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        nneighbors = []
        fneighbors = []
    else:
        key_ = 'ts_org'
        include_neighbors = False

    ptr = 0
    for batch in dataloader:
        ts = batch[key_]
        if ts.ndim == 3:
            bs, w, h = ts.shape
        else:
            bs, w = ts.shape
            h =1

        if isinstance(ts, np.ndarray):
            ts = torch.from_numpy(ts).float()
            targets.append(torch.as_tensor(batch['target'], device=next(model.parameters()).device))
        else:
            ts = ts.to(device, non_blocking=True)
            targets.append(batch['target'].to(device, non_blocking=True))
            
        res = model(ts.view(bs, h, w), forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr: ptr+bs] = res['features'].cpu()
            ptr += bs
        
        predictions.append(torch.argmax(output, dim=1))
        probs.append(F.softmax(output, dim=1))

        if include_neighbors:
            nneighbors.append(batch['possible_nneighbors'])
            fneighbors.append(batch['possible_fneighbors'])
    
    predictions = torch.cat(predictions, dim=0)
    probs = torch.cat(probs, dim=0)
    targets = torch.cat(targets, dim=0)

    if include_neighbors:
        nneighbors = torch.cat(nneighbors, dim=0)
        fneighbors = torch.cat(fneighbors, dim=0)
        out = {'predictions': predictions, 'probabilities': probs, 'targets': targets, 'neighbors': nneighbors, 'fneighbors': fneighbors}

    else:
        out = {'predictions': predictions, 'probabilities': probs, 'targets': targets}

    if return_features:
        feat_np = features.numpy()  # save features in csv
        fhdr = [str(x) for x in range(feat_np.shape[1])] + ['Class']
        
        feat_np = np.hstack((feat_np, np.array(targets.cpu().numpy())[np.newaxis].T)) 

        feat_df = pd.DataFrame(feat_np, columns=fhdr)

        prob_np = out['probabilities'].cpu().numpy()
        phdr = [str(x) for x in range(prob_np.shape[1])] + ['Class']
        
        prob_np = np.hstack((prob_np, np.array(targets.cpu().numpy())[np.newaxis].T)) 
        prob_df = pd.DataFrame(prob_np, columns=phdr)

        if is_training:
            feat_df.to_csv(p['classification_trainfeatures'], index=False, header=True, sep=',')
            prob_df.to_csv(p['classification_trainprobs'], index=False, header=True, sep=',')
        else:
            feat_df.to_csv(p['classification_testfeatures'], index=False, header=True, sep=',')
            prob_df.to_csv(p['classification_testprobs'], index=False, header=True, sep=',')

        return out, features.cpu()

    else:
        return out


@torch.no_grad()
def classification_evaluate(predictions, entropy_weight=2, consistency_weight=1, inconsistency_weight=0):
    # Evaluate model based on classification loss.
    device = predictions['predictions'].device

    # Neighbors and anchors
    probs = predictions['probabilities'].to(device, non_blocking=True)
    neighbors = predictions['neighbors'].to(device, non_blocking=True)
    fneighbors = predictions['fneighbors'].to(device, non_blocking=True)
    org_anchors = torch.arange(neighbors.size(0), device=device).view(-1,1).expand_as(neighbors)

    # Entropy loss
    entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

    # Consistency loss
    similarity = torch.matmul(probs, probs.t())
    neighbors = neighbors.contiguous().view(-1)
    anchors = org_anchors.contiguous().view(-1)
    similarity_n = similarity[anchors, neighbors]
    ones = torch.ones_like(similarity_n)
    consistency_loss = F.binary_cross_entropy(similarity_n, ones).item()


    fneighbors = fneighbors.contiguous().view(-1)
    # anchors = org_anchors.contiguous().view(-1)
    similarity_fn = similarity[anchors, fneighbors]
    ones = torch.ones_like(similarity_fn)
    inconsistency_loss = F.binary_cross_entropy(similarity_fn, ones).item()

    # Total loss #TODO: check loss weights
    total_loss = consistency_weight*consistency_loss - entropy_weight*entropy_loss - inconsistency_weight*inconsistency_loss

    output = {'entropy': entropy_loss, 'consistency': consistency_loss, 'inconsistency': inconsistency_loss, 'total_loss': total_loss}

    return {'classification': output}


@torch.no_grad()
def pr_evaluate(all_predictions, class_names=None, majority_label=0, logger=None):
    
    logger = EmptyLogger() if logger is None else logger

    head = all_predictions
    targets = head['targets']
    probs = head['probabilities']

    scores = 1-probs[:, majority_label]

    precision, recall, thresholds = precision_recall_curve(scores, targets, task='binary')
    
    try:
        f1_score = 2*precision*recall / (precision+recall)
        if torch.isnan(f1_score).any():
            f1_score = torch.nan_to_num(f1_score)
            logger.log('f1: Nan --> 0')     
    except ZeroDivisionError:
        f1_score = [0.0]
        logger.log('f1: 0 --> 0')

    best_f1_index = torch.argmax(f1_score)

    rep_f1 = f1_score[best_f1_index]

    if class_names=='Anom':
        best_threshold = thresholds[best_f1_index]
        anomalies = [1 if s >= best_threshold else 0 for s in scores]
        best_tn, best_fp, best_fn, best_tp = confusion_matrix(targets, anomalies).ravel()
        logger.log(f"Anomalies --> TP: {best_tp}, TN: {best_tn}, FN: {best_fn}, FP: {best_fp}")
        logger.log(f"Mazority label: {majority_label}")
        logger.log(metrics.classification_report(targets, anomalies))

    return rep_f1
