from sklearn.metrics import precision_recall_curve

from metrics.f1_score_f1_pa import get_adjust_F1PA, get_accuracy_precision_recall_fscore, event_f1
from metrics.Matthews_correlation_coefficient import MCC
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.vus.metrics import get_range_vus_roc
import numpy as np
import matplotlib
matplotlib.use('agg',force=True)
import matplotlib.pyplot as plt

def combine_all_evaluation_scores(pred_labels, y_test, window_size: int = 100):
    events_pred = convert_vector_to_events(pred_labels) 
    events_gt = convert_vector_to_events(y_test)
    Trange = (0, len(y_test))
    f1_event = event_f1(y_test, events_gt, pred_labels)
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    MCC_score = MCC(y_test, pred_labels)
    vus_results = get_range_vus_roc(pred_labels, y_test, window_size)
    accuracy, precision, recall, f_score, f05_score, tp, fp, fn, tn = get_accuracy_precision_recall_fscore(y_test, pred_labels)
    
    # change predictions to point adjasment and get metrics again
    pa_accuracy, pa_precision, pa_recall, pa_f_score, latency = get_adjust_F1PA(pred_labels, y_test)
    events_pred = convert_vector_to_events(pred_labels) 
    events_gt = convert_vector_to_events(y_test)
    pa_f1_event = event_f1(y_test, events_gt, pred_labels)
    pa_affiliation = pr_from_events(events_pred, events_gt, Trange)
    pa_MCC_score = MCC(y_test, pred_labels)
    pa_vus_results = get_range_vus_roc(pred_labels, y_test, 100) # default slidingWindow = 100

    score_list = {"Affiliation precision": affiliation['precision'], 
                  "Affiliation recall": affiliation['recall'],
                  "Event F1": f1_event,
                  "MCC": MCC_score,
                  "R_AUC_ROC": vus_results["R_AUC_ROC"], 
                  "R_AUC_PR": vus_results["R_AUC_PR"],
                  "VUS_ROC": vus_results["VUS_ROC"],
                  "VUS_PR": vus_results["VUS_PR"],
                  "accuracy": accuracy,
                  "precision": precision,
                  "recall": recall,
                  "f_score": f_score,
                  "f05_score": f05_score,
                  "tp": tp,
                  "fp": fp,
                  "fn": fn,
                  "tn": tn,
                  "pa_Affiliation precision": pa_affiliation['precision'], 
                  "pa_Affiliation recall": pa_affiliation['recall'],
                  "pa_Event F1": pa_f1_event,
                  "pa_MCC": pa_MCC_score,
                  "pa_R_AUC_ROC": pa_vus_results["R_AUC_ROC"], 
                  "pa_R_AUC_PR": pa_vus_results["R_AUC_PR"],
                  "pa_VUS_ROC": pa_vus_results["VUS_ROC"],
                  "pa_VUS_PR": pa_vus_results["VUS_PR"],
                  "pa_accuracy": pa_accuracy,
                  "pa_precision": pa_precision,
                  "pa_recall": pa_recall,
                  "pa_f_score": pa_f_score,
                  "pa_latency": latency
    }

    return score_list

def evaluate(logger, epoch, energy, input, gt, start, end, tag, ch=None, threshold=None, det_threshold=1, pre_classify=True, make_figures=False):
    """Evaluate model and make images.
    Model need to output propabilities for each time step along with how these propabilites are maped to the whole time series.
    Also each detection is added only to the last checking window.
    The user can determine a predefined threshold that is applied before or after accomulation of the predictions in the time series.
    Args:
        epoch (int): the epoch number for logging purposes
        energy (torch.Tensor): transformer output energy or propabilities of shape (N, Window Size) or (N, Window Size, D)
        input (np.ndarray): input data of shape (N, D)
        gt (np.ndarray): ground truth labels of shape (N, 1) or (N, D)
        start (np.ndarray): start indices of the windows
        end (np.ndarray): end indices of the windows
        tag (str): tag for logging purposes
        ch (int, optional): window length to accomodate the detections. Defaults to window size of the model.
        threshold (float, optional): threshold for classification. Defaults to best threshold for regular F1 score.
        det_threshold (int, optional): if pre_classify is True, a detection is true if the accomulated detections are higher than this threshold. Defaults to 1.
        pre_classify (bool, optional): whether to classify predictions before accumulation. Defaults to True.
        make_figures (bool, optional): whether to make figures. Defaults to False.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation scores.
        threshold (float): the threshold used for classification.
    """
    ch = end[0] - start[0] if ch is None else ch
    
    if threshold is None and pre_classify:
        acc_prob = np.zeros((end[-1] - start[0], 1)) #, pred.shape[-1])) check if shape.size >=3 then shape[-1]
        for e, a in zip(end, energy):
            acc_prob[e-ch:e] += a.reshape(-1,1)[-ch:]
        acc_prob /= (ch/(start[1] - start[0]))
        precision, recall, thresholds = precision_recall_curve(gt, acc_prob, pos_label=1)
        f1_score = (2 * precision * recall) / (precision + recall)
        if np.isnan(f1_score).any():
            f1_score = np.nan_to_num(f1_score)
        best_index = np.argmax(f1_score)
        threshold = thresholds[best_index]
        
    pred = (energy > threshold).astype(int) if pre_classify else energy
    
    acc_prob = np.zeros((end[-1] - start[0], 1)) #check if shape.size >=3 then shape[-1]
    for e, a in zip(end, pred):
        acc_prob[e-ch:e] += a.reshape(-1,1)[-ch:]
    acc_prob /= (ch/start[1] - start[0])

    if ((threshold is None) and (not pre_classify)) or (pre_classify and det_threshold is None):
        precision, recall, thresholds = precision_recall_curve(gt, acc_prob, pos_label=1)
        f1_score = (2 * precision * recall) / (precision + recall)
        if np.isnan(f1_score).any():
            f1_score = np.nan_to_num(f1_score)
        best_index = np.argmax(f1_score)
        threshold = thresholds[best_index]
        det_threshold = thresholds[best_index]
    
    print(f"{tag}Threshold :", threshold)
    
    assert acc_prob.shape[1] == gt.shape[1] == 1, "predictions and gt should have shape (N, 1), Multi-dimensional predictions and gt are not supported yet."
    acc_prob = acc_prob[:, 0] # flatten to shape (N,)
    gt = gt[:, 0] # flatten to shape (N,)
    if make_figures:
        figures(logger, inputs=input, labels=gt, predictions=acc_prob, mode=f"{tag}", epoch=epoch)
    
    acc_prob = (acc_prob >= det_threshold).astype(int) if pre_classify else (acc_prob > threshold).astype(int) #TODO maybe divide by the window length and keep it as float
    
    eval_scores = combine_all_evaluation_scores(acc_prob, gt)
    if make_figures:
        figures(logger, inputs=input, labels=gt, predictions=acc_prob, mode=f"{tag}", epoch=epoch, pa="_pa")

    return eval_scores, threshold

def evaluate_all_thresholds(energy, input, gt, start, end, tag="final", ch=None, threshold=None, pre_classify=True):
    ch = end[0] - start[0] if ch is None else ch
    
    if threshold is None and pre_classify:
        acc_prob = np.zeros((end[-1] - start[0], 1)) #, pred.shape[-1])) check if shape.size >=3 then shape[-1]
        for e, a in zip(end, energy):
            acc_prob[e-ch:e] += a.reshape(-1,1)[-ch:]
        acc_prob /= (ch/(start[1] - start[0]))
        precision, recall, thresholds = precision_recall_curve(gt, acc_prob, pos_label=1)
        f1_score = (2 * precision * recall) / (precision + recall)
        if np.isnan(f1_score).any():
            f1_score = np.nan_to_num(f1_score)
        best_index = np.argmax(f1_score)
        threshold = thresholds[best_index]

    pred = (energy > threshold).astype(int) if pre_classify else energy
    
    acc_prob = np.zeros((end[-1] - start[0], 1)) #, pred.shape[-1])) check if shape.size >=3 then shape[-1]
    for e, a in zip(end, pred):
        acc_prob[e-ch:e] += a.reshape(-1,1)[-ch:]
    acc_prob /= (ch/(start[1] - start[0]))

    _, _, thresholds = precision_recall_curve(gt, acc_prob, pos_label=1)
    thresholds_int = (thresholds*ch).astype(int)
    if pre_classify:
        thresholds_int, idxs = np.unique(thresholds_int, return_index=True)
        thresholds = thresholds[idxs]
    else:
        thresholds = thresholds[::len(thresholds)//(ch+1)]
        thresholds_int = thresholds_int[::len(thresholds)//(ch+1)]

    assert acc_prob.shape[1] == gt.shape[1] == 1, "predictions and gt should have shape (N, 1), Multi-dimensional predictions and gt are not supported yet."
    acc_prob = acc_prob[:, 0] # flatten to shape (N,)
    gt = gt[:, 0] # flatten to shape (N,)

    for th, th_int in zip(thresholds, thresholds_int):
        identification = th_int if pre_classify else th

        curr_preds = (acc_prob >= th).astype(int)
        figures(inputs=input, labels=gt, predictions=curr_preds, mode=f"{tag}Energy", epoch=identification)
    
        eval_scores = combine_all_evaluation_scores(curr_preds, gt)
        figures(inputs=input, labels=gt, predictions=curr_preds, mode=f"{tag}Energy", epoch=identification, pa="_pa")

    return eval_scores, threshold

def figures(logger, inputs, labels, predictions, mode="Combined", epoch=0, pa=""):
    """
    inputs: np.ndarray of shape (N, 38)
    labels: np.ndarray of shape (N,)
    predictions: np.ndarray of shape (N,)
    mode: str, either "Train" or "Combined"
    epoch: int
    """
    n_samples, n_features = inputs.shape
    x = np.arange(n_samples)

    i = 0
    # for i in range(n_features):
    with plt.ioff():
        fig, ax = plt.subplots(figsize=(12, 6))

        y = inputs[:, i]
        ax.plot(x, y, color="black", linewidth=1, label=f"feature {i}")

        # Shade where labels == 1
        ax.fill_between(
            x,
            y.min()-1,
            (y.max()-y.min()+1)/2,
            where=labels[:].astype(bool), #If we have more than one label we can go for label per feature
            color="red",
            alpha=0.2,
            step="mid",
            label="label=1",
        )

        pred_unique = np.unique(predictions)
        
        if set(pred_unique).issubset({0, 1}):
            # Mode 1: Binary
            normalized = predictions.astype(float)
        elif predictions.dtype in [np.float32, np.float64] and \
            0 <= predictions.min() and predictions.max() <= 1:
            # Mode 2: Float [0, 1]
            normalized = predictions
        else:
            # Mode 3: Integer range
            normalized = predictions.astype(float) / predictions.max()
        
        # Create 2D grid for pcolormesh gradient
        y_edges = np.linspace((y.max()-y.min()-1)/2, y.max()+1, 2)
        x_edges = np.concatenate([[x[0]], (x[:-1] + x[1:]) / 2, [x[-1]]])
        X, Y = np.meshgrid(x_edges, y_edges)
            
        ax.pcolormesh(X, Y, normalized[np.newaxis, :], cmap="Blues",
                        alpha=1, shading="flat", vmin=0, vmax=1)

        ax.set_title(f"Feature {i}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right")

        logger.add_figure(f"{mode}/feature{pa}", fig, step=epoch)
        plt.close(fig)
    return

if __name__ == '__main__':
    y_test = np.load("data/events_pred_MSL.npy")+0
    pred_labels = np.load("data/events_gt_MSL.npy")+0
    anomaly_scores = np.load("data/events_scores_MSL.npy")
    print(len(y_test), max(anomaly_scores), min(anomaly_scores))
    score_list_simple = combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores)

    for key, value in score_list_simple.items():
        print('{0:21} :{1:10f}'.format(key, value))