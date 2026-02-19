import numpy as np
import pandas as pd
import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
import warnings
from utils.utils import mkdir_if_missing
import matplotlib.pyplot as plt
import argparse

warnings.filterwarnings("ignore", category=RuntimeWarning)

def adjust_predicts(label, predict=None, calc_latency=False):
    
    label = np.asarray(label)
    latency = 0
    
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(actual)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
        
    MCM = metrics.multilabel_confusion_matrix(actual, predict, labels = [1, 0])

    pa_tn = MCM[0][0, 0]
    pa_tp = MCM[0][1, 1]
    pa_fp = MCM[0][0, 1]
    pa_fn = MCM[0][1, 0]
        
    prec = pa_tp / (pa_tp + pa_fp)
    rec = pa_tp / (pa_tp + pa_fn)
    f1_score = 2 * (prec * rec) / (prec + rec)
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4), pa_tp, pa_tn, pa_fp, pa_fn, prec , rec, f1_score
    else:
        return predict, prec, rec, f1_score, pa_tp, pa_tn, pa_fp, pa_fn
    
def add_summary_statistics(res_df):
    # Compute the sum of 'best_tp', 'best_tn', 'best_fp', 'best_fn'
    sum_best_tp = res_df['best_tp'].sum()
    sum_best_tn = res_df['best_tn'].sum()
    sum_best_fp = res_df['best_fp'].sum()
    sum_best_fn = res_df['best_fn'].sum()

    # Calculate precision, recall and f1 score
    precision = sum_best_tp / (sum_best_tp + sum_best_fp) if (sum_best_tp + sum_best_fp) > 0 else 0
    recall = sum_best_tp / (sum_best_tp + sum_best_fn) if (sum_best_tp + sum_best_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate the average and std of 'roc' and 'pr'
    roc_avg = res_df['roc'].mean()
    pr_avg = res_df['pr'].mean()
    f1_avg = res_df['b_f_1'].mean()
    prec_avg = res_df['best_pre'].mean()
    rec_avg = res_df['best_rec'].mean()

    roc_std = res_df['roc'].std()
    pr_std = res_df['pr'].std()
    f1_std = res_df['b_f_1'].std()
    prec_std = res_df['best_pre'].std()
    rec_std = res_df['best_rec'].std()


    # Append the results to the dataframe
    summary_row = pd.Series({
        'name': 'sum',
        'best_tp': sum_best_tp,
        'best_tn': sum_best_tn,
        'best_fp': sum_best_fp,
        'best_fn': sum_best_fn,
        'best_pre': precision,
        'best_rec': recall,
        'b_f_1': f1_score,
        'roc': roc_avg,
        'pr': pr_avg
    })

    std_row = pd.Series({
        'name': 'std',
        'roc': roc_std,
        'pr': pr_std,
        'b_f_1': f1_std,
        'best_pre': prec_std,
        'best_rec': rec_std
    })

    mean_row = pd.Series({
        'name': 'mean',
        'roc': roc_avg,
        'pr': pr_avg,
        'b_f_1': f1_avg,
        'best_pre': prec_avg,
        'best_rec': rec_avg
    })

    # Append the rows to the dataframe
    res_df = res_df._append(summary_row, ignore_index=True)
    res_df = res_df._append(mean_row, ignore_index=True)
    res_df = res_df._append(std_row, ignore_index=True)
    
    return res_df

def add_summary_statistics_pa(res_df):
    # Compute the sum of 'best_tp', 'best_tn', 'best_fp', 'best_fn'
    sum_pa_tp = res_df['pa_tp'].sum()
    sum_pa_tn = res_df['pa_tn'].sum()
    sum_pa_fp = res_df['pa_fp'].sum()
    sum_pa_fn = res_df['pa_fn'].sum()

    # Calculate precision, recall and f1 score
    precision = sum_pa_tp / (sum_pa_tp + sum_pa_fp) if (sum_pa_tp + sum_pa_fp) > 0 else 0
    recall = sum_pa_tp / (sum_pa_tp + sum_pa_fn) if (sum_pa_tp + sum_pa_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    # Append the results to the dataframe
    summary_row = pd.Series({
        'pa_tp': sum_pa_tp,
        'pa_tn': sum_pa_tn,
        'pa_fp': sum_pa_fp,
        'pa_fn': sum_pa_fn,
        'pa_pre': precision,
        'pa_rec': recall,
        'pa_f1': f1_score,
    })


    # Append the row to the dataframe
    res_df = res_df._append(summary_row, ignore_index=True)
    
    return res_df

def main(args):

    res_df = pd.DataFrame(columns=['name', 'tp', 'tn', 'fp', 'fn', 'roc', 'pr', 
                               'best_tp', 'best_tn', 'best_fp', 'best_fn', 'best_pre', 'best_rec', 'b_f_1']) 

    pa_df = pd.DataFrame(columns=['name', 'pa_tp', 'pa_tn', 'pa_fp', 'pa_fn', 'pa_pre', 'pa_rec', 'pa_f1', 'latency'])
    
    version = args.version
    save_dir = args.save_dir
    mkdir_if_missing(save_dir)
    save_dir = f'_{save_dir}' if save_dir is not None else ''
    
    folder = version #"big_train_custom_lr" #args.version

    data_info = os.listdir(f'results/smd/{folder}')
    files = [file for file in data_info if file.startswith('machine-')]
    files = sorted(files)
    print(files)
    ds_name = 'smd'
    for filename in files: #['GECCO']: #data_info['chan_id']: #files: #['M-6']: #data_info['chan_id']:
        if filename!='.json': # and 'real_' in filename:
            print(filename)
            df_train = pd.read_csv(f"results/{ds_name}/{folder}/{filename}/classification{save_dir}/classification_trainprobs.csv")
            df_test = pd.read_csv(f"results/{ds_name}/{folder}/{filename}/classification{save_dir}/classification_testprobs.csv")
            cl_num = df_train.shape[1] - 1

            df_train['Class'] = np.where((df_train['Class'] == 0), 0, 1)
            df_train['pred']=df_train[df_train.columns[0:cl_num]].idxmax(axis=1)

            score_col = df_train['pred'].value_counts().idxmax()
            
            df_test['Class'] = np.where((df_test['Class'] == 0), 0, 1)
            df_test['pred'] = df_test[df_test.columns[0:cl_num]].idxmax(axis=1)
            
    #         score_col = df_test['pred'].value_counts().idxmax()
            
            roc_auc, pr_auc, best_tn, best_tp, best_fp, best_fn, best_pre, best_rec, best_f1 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            try:

                df_test['pred'] = np.where((df_test['pred'] == score_col), 0, 1)

                MCM = metrics.multilabel_confusion_matrix(df_test['Class'], df_test['pred'], labels = [1, 0])
                tn = MCM[0][0, 0]
                tp = MCM[0][1, 1]
                fp = MCM[0][0, 1]
                fn = MCM[0][1, 0]

                pre=tp/(tp+fp)
                recall = tp/(tp+fn)
                f_1 = 2*pre*recall/(pre+recall)

                scores = 1-df_test[score_col]
                # Calculate AU-ROC
                roc_auc = roc_auc_score(df_test['Class'], scores)
                print('AU-ROC : ', roc_auc)

                # Calculate AU-PR
                pr_auc = average_precision_score(df_test['Class'], scores)
                print('AU-PR : ', pr_auc)

                fpr, tpr, thresholds = roc_curve(df_test['Class'], scores, pos_label=1)
                plt.figure(figsize=(5, 5))
                plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (area = {roc_auc:.2f})")
                plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Classifier")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Receiver Operating Characteristic (ROC) Curve")
                plt.legend(loc="lower right")
                # plt.grid(True)
                mkdir_if_missing(f'results/{ds_name}/{folder}/{save_dir}/roc_curves')
                plt.savefig(f'results/{ds_name}/{folder}/{save_dir}/roc_curves/roc_{filename.split(".")[0]}.png')

                precision, recall, thresholds = precision_recall_curve(df_test['Class'], scores, pos_label=1)
                plt.figure(figsize=(5, 5))
                plt.plot(recall, precision, color="darkorange", label=f"PR curve (area = {pr_auc:.2f})")
                plt.plot([0, 1], [0.5, 0.5], color="navy", linestyle="--", label="Random Classifier")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision/Recall Curve")
                plt.legend(loc="lower right")
                # plt.grid(True)
                mkdir_if_missing(f'results/{ds_name}/{folder}/{save_dir}/pr_rec_curves')
                plt.savefig(f'results/{ds_name}/{folder}/{save_dir}/pr_rec_curves/pr_{filename.split(".")[0]}.png')

                res = pd.DataFrame()
                res['pre'] = precision
                res['rec'] = recall
                res['f1'] = 2*res['pre']*res['rec'] / (res['pre']+res['rec'])
                best_idx = res['f1'].argmax()
                best_f1 = res['f1'][best_idx]
                best_pre = res['pre'][best_idx]
                best_rec = res['rec'][best_idx]
                best_thr = thresholds[best_idx]
                print('Best f1 : ', best_f1, 'best_thr', best_thr)
                anomalies = [True if s >= best_thr else False for s in scores]

                best_tn, best_fp, best_fn, best_tp = confusion_matrix(df_test['Class'], anomalies).ravel()
            except ValueError:
                pass

            new_row = pd.Series([filename, tp, tn, fp, fn, roc_auc, pr_auc, best_tp, best_tn, best_fp, best_fn, best_pre, best_rec, best_f1],
                                    index=['name', 'tp', 'tn', 'fp', 'fn', 'roc', 'pr', 'best_tp', 'best_tn', 'best_fp', 'best_fn', 'best_pre', 'best_rec', 'b_f_1'])
            res_df = res_df._append(new_row, ignore_index=True)
            
            
            # pa_f1 = -1
            # for thr in thresholds:
            #     preds_pa = [True if s >= thr else False for s in scores]
            #     pa_prediction, t_latency, t_tp, t_tn, t_fp, t_fn, t_pre, t_rec, t_f1 = adjust_predicts(df_test['Class'], preds_pa, True)
            #     if t_f1 > pa_f1:
            #         latency, pa_tp, pa_tn, pa_fp, pa_fn, pa_pre, pa_rec, pa_f1 = t_latency, t_tp, t_tn, t_fp, t_fn, t_pre, t_rec, t_f1
                    
            # new_row1 = pd.Series([filename, pa_tp, pa_tn, pa_fp, pa_fn, pa_pre, pa_rec, pa_f1, latency],
            #                         index=['name', 'pa_tp', 'pa_tn', 'pa_fp', 'pa_fn', 'pa_pre', 'pa_rec', 'pa_f1', 'latency'])   
            # pa_df = pa_df._append(new_row1, ignore_index=True)
            
        
    res_df = add_summary_statistics(res_df)
    res_df.to_csv(f'results/{ds_name}/{folder}/{save_dir}/results_woincon.csv')

    # pa_df = add_summary_statistics_pa(pa_df)
    # pa_df.to_csv('smd_5_results_pa.csv')


if __name__ == "__main__":
    FLAGS = argparse.ArgumentParser(description='classification Loss')
    FLAGS.add_argument('--version', help='Experiment version', type=str)
    FLAGS.add_argument('--save_dir', help='Save directory', type=str)
    args = FLAGS.parse_args()
    main(args)

