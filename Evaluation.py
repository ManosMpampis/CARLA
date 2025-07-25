import os
import warnings

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
warnings.filterwarnings("ignore", category=RuntimeWarning)

from utils.utils import log

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
        return predict, prec, rec, f1_score, pa_tp, pa_tn, pa_fp, pa_fn,

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
    roc_std = res_df['roc'].std()
    pr_avg = res_df['pr'].mean()
    pr_std = res_df['pr'].std()

    # Append the results to the dataframe
    summary_row = pd.Series({
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
        'roc': roc_std,
        'pr': pr_std
    })

    # Append the rows to the dataframe
    res_df = res_df._append(summary_row, ignore_index=True)
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

if __name__ == "__main__":
    res_df = pd.DataFrame(columns=['name', 'tp', 'tn', 'fp', 'fn', 'roc', 'pr', 
                               'best_tp', 'best_tn', 'best_fp', 'best_fn', 'best_pre', 'best_rec', 'b_f_1']) 

    pa_df = pd.DataFrame(columns=['name', 'pa_tp', 'pa_tn', 'pa_fp', 'pa_fn', 'pa_pre', 'pa_rec', 'pa_f1', 'latency'])


    # with open('../../datasets/MSL_SMAP/labeled_anomalies.csv', 'r') as file:
    #     csv_reader = pd.read_csv(file, delimiter=',')

    # data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']


    # data_info = os.listdir('../datasets/KPI/train/')

    data_info = os.listdir(os.path.join('../../datasets', 'UCR'))  
    data_info = sorted(data_info)

    # data_info = os.listdir('../../datasets/SMD/train/')
    # files = [file for file in data_info if file.startswith('machine-')]
    # files = sorted(files)
    ds_name = 'ucr'

    global verbose, file_path
    verbose = 2
    file_path = f"results/{ds_name}/Evaluation_notebook.txt"

    for filename in data_info: #['GECCO']: #data_info['chan_id']: #files: #['M-6']: #data_info['chan_id']:
        if filename!='.json': # and 'real_' in filename:
            log(filename, verbose=verbose, file_path=file_path)
            df_train = pd.read_csv(f"results/{ds_name}/" + filename + "/classification/classification_trainprobs.csv")
            df_test = pd.read_csv(f"results/{ds_name}/" + filename + "/classification/classification_testprobs.csv")
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
                log(f'AU-ROC : {roc_auc}', verbose=verbose, file_path=file_path)

                # Calculate AU-PR
                pr_auc = average_precision_score(df_test['Class'], scores)
                log(f'AU-PR : {pr_auc}', verbose=verbose, file_path=file_path)

                fpr, tpr, thresholds = roc_curve(df_test['Class'], scores, pos_label=1)
                precision, recall, thresholds = precision_recall_curve(df_test['Class'], scores, pos_label=1)


                res = pd.DataFrame()
                res['pre'] = precision
                res['rec'] = recall
                res['f1'] = 2*res['pre']*res['rec'] / (res['pre']+res['rec'])
                best_idx = res['f1'].argmax()
                best_f1 = res['f1'][best_idx]
                best_pre = res['pre'][best_idx]
                best_rec = res['rec'][best_idx]
                best_thr = thresholds[best_idx]
                log(f'Best f1 : {best_f1} | Best threshold : {best_thr}', verbose=verbose, file_path=file_path)
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
    res_df.to_csv(f'{ds_name}_results_woincon.csv')

    # pa_df = add_summary_statistics_pa(pa_df)
    # pa_df.to_csv('smd_5_results_pa.csv')