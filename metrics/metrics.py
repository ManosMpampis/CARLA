from metrics.f1_score_f1_pa import get_adjust_F1PA, get_accuracy_precision_recall_fscore, event_f1
from metrics.Matthews_correlation_coefficient import MCC
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
from metrics.vus.metrics import get_range_vus_roc
import numpy as np

def combine_all_evaluation_scores(pred_scores, y_test, threshold, window_size: int = 100):
    """Threshold scores for decisions; retain continuous scores for AUC/VUS.

    PA adjusts only the locally created predictions. Score-based metrics are
    unchanged by PA and are exposed under both ordinary and PA result keys.
    """
    pred_scores = np.asarray(pred_scores)
    pred_labels = (pred_scores >= threshold).astype(np.int32)
    
    events_pred = convert_vector_to_events(pred_labels) 
    events_gt = convert_vector_to_events(y_test)
    Trange = (0, len(y_test))
    f1_event = event_f1(y_test, events_gt, pred_labels)
    affiliation = pr_from_events(events_pred, events_gt, Trange)
    MCC_score = MCC(y_test, pred_labels)
    vus_results = get_range_vus_roc(pred_scores, y_test, window_size)
    accuracy, precision, recall, f_score, f05_score, tp, fp, fn, tn = get_accuracy_precision_recall_fscore(y_test, pred_labels)
    
    # change predictions to point adjasment and get metrics again
    pa_accuracy, pa_precision, pa_recall, pa_f_score, latency = get_adjust_F1PA(pred_labels, y_test)
    events_pred = convert_vector_to_events(pred_labels) 
    events_gt = convert_vector_to_events(y_test)
    pa_f1_event = event_f1(y_test, events_gt, pred_labels)
    pa_affiliation = pr_from_events(events_pred, events_gt, Trange)
    pa_MCC_score = MCC(y_test, pred_labels)

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
                  "f1_score": f_score,
                  "f05_score": f05_score,
                  "tp": tp,
                  "fp": fp,
                  "fn": fn,
                  "tn": tn,
                  "pa_Affiliation precision": pa_affiliation['precision'], 
                  "pa_Affiliation recall": pa_affiliation['recall'],
                  "pa_Event F1": pa_f1_event,
                  "pa_MCC": pa_MCC_score,
                  "pa_accuracy": pa_accuracy,
                  "pa_precision": pa_precision,
                  "pa_recall": pa_recall,
                  "pa_f1_score": pa_f_score,
                  "pa_latency": latency
    }

    return score_list
