import os
import warnings

import numpy as np
import pandas as pd
import torch

from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
warnings.filterwarnings("ignore", category=RuntimeWarning)

from utils.utils import Logger

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

    # Compute the sum of 'best_tp_train', 'best_tn_train', 'best_fp_train', 'best_fn_train'
    sum_best_tp_train = res_df['best_tp_train'].sum()
    sum_best_tn_train = res_df['best_tn_train'].sum()
    sum_best_fp_train = res_df['best_fp_train'].sum()
    sum_best_fn_train = res_df['best_fn_train'].sum()

    # Calculate precision, recall and f1 score
    precision = sum_best_tp / (sum_best_tp + sum_best_fp) if (sum_best_tp + sum_best_fp) > 0 else 0
    recall = sum_best_tp / (sum_best_tp + sum_best_fn) if (sum_best_tp + sum_best_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    precision_train = sum_best_tp_train / (sum_best_tp_train + sum_best_fp_train) if (sum_best_tp_train + sum_best_fp_train) > 0 else 0
    recall_train = sum_best_tp_train / (sum_best_tp_train + sum_best_fn_train) if (sum_best_tp_train + sum_best_fn_train) > 0 else 0
    f1_score_train = 2 * (precision_train * recall_train) / (precision_train + recall_train) if (precision_train + recall_train) > 0 else 0

    # Calculate the average and std of 'roc' and 'pr'
    roc_avg = res_df['roc'].mean()
    roc_std = res_df['roc'].std()
    pr_avg = res_df['pr'].mean()
    pr_std = res_df['pr'].std()

    # Append the results to the dataframe
    summary_row = pd.Series({
        'name': 'Best Test Threshold/Train Threshold values',
        'best_tp': sum_best_tp,
        'best_tn': sum_best_tn,
        'best_fp': sum_best_fp,
        'best_fn': sum_best_fn,
        'best_tp_train': sum_best_tp_train,
        'best_tn_train': sum_best_tn_train,
        'best_fp_train': sum_best_fp_train,
        'best_fn_train': sum_best_fn_train,
        'best_pre': precision,
        'best_rec': recall,
        'b_f_1': f1_score,
        'best_pre_train': precision_train,
        'best_rec_train': recall_train,
        'b_f_1_train': f1_score_train,
        'roc': roc_avg,
        'pr': pr_avg
    })

    std_row = pd.Series({
        'name': 'Std of roc and pr',
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
    columns_names = ['name', 'tp', 'tn', 'fp', 'fn', 'roc', 'pr', 
                     'best_tp', 'best_tn', 'best_fp', 'best_fn', 'best_pre', 'best_rec', 'b_f_1',
                     'best_tp_train', 'best_tn_train', 'best_fp_train', 'best_fn_train', 'best_pre_train', 'best_rec_train', 'b_f_1_train']
    data_list = []
    # res_df = pd.DataFrame(columns=columns_names) 

    # pa_df = pd.DataFrame(columns=['name', 'pa_tp', 'pa_tn', 'pa_fp', 'pa_fn', 'pa_pre', 'pa_rec', 'pa_f1', 'latency'])
    database = 'SMD'
    database = database.lower()
    version = 'temp'

    result_file_path = os.path.join(os.path.dirname(__file__), 'results', database, version)
    data_info = os.listdir(result_file_path)
    files = [ dir_name for dir_name in data_info if os.path.isdir(os.path.join(result_file_path, dir_name))]
    files = sorted(files)

    logger = Logger(version=version, verbose=2, file_path=result_file_path, use_tensorboard=False, file_name='Evaluation_logs.txt')


    for filename in files: #['GECCO']: #data_info['chan_id']: #files: #['M-6']: #data_info['chan_id']:
        if filename!='.json': # and 'real_' in filename:
            logger.log(filename)

            experiment_folder = f"results/{database}/{version}/{filename}/classification"

            df_train = pd.read_csv(os.path.join(experiment_folder, 'classification_trainprobs.csv'))
            df_test = pd.read_csv(os.path.join(experiment_folder, 'classification_testprobs.csv'))
            majority_class = torch.load(os.path.join(experiment_folder, 'model.pth.tar'))['majority_label'].item()

            cl_num = df_train.shape[1] - 1

            df_train['Class'] = np.where((df_train['Class'] == 0), 0, 1)
            df_train['pred']=df_train[df_train.columns[0:cl_num]].idxmax(axis=1)

            score_col = df_train['pred'].value_counts().idxmax()
            score_col = str(majority_class)
            train_scores = 1-df_train[score_col]
            train_precision, train_recall, train_thresholds = precision_recall_curve(df_train['Class'], train_scores, pos_label=1)
            train_f1_scores = 2 * train_precision * train_recall / (train_precision + train_recall)

            best_train_idx = train_f1_scores.argmax()
            best_train_threshold = train_thresholds[best_train_idx]


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
                logger.log(f'AU-ROC : {roc_auc}')

                # Calculate AU-PR
                pr_auc = average_precision_score(df_test['Class'], scores)
                logger.log(f'AU-PR : {pr_auc}')

                fpr, tpr, thresholds = roc_curve(df_test['Class'], scores, pos_label=1)
                precision, recall, thresholds = precision_recall_curve(df_test['Class'], scores, pos_label=1)


                res = pd.DataFrame()
                res['pre'] = precision
                res['rec'] = recall
                res['f1'] = 2*precision*recall / (precision+recall)
                best_idx = res['f1'].argmax()
                best_f1 = res['f1'][best_idx]
                best_pre = res['pre'][best_idx]
                best_rec = res['rec'][best_idx]
                best_thr = thresholds[best_idx]
                logger.log(f'Best f1 : {best_f1} | Best threshold : {best_thr}')
                anomalies = [True if s >= best_thr else False for s in scores]

                best_tn, best_fp, best_fn, best_tp = confusion_matrix(df_test['Class'], anomalies).ravel()

                train_anom = [True if s >= best_train_threshold else False for s in scores]
                best_tn_train, best_fp_train, best_fn_train, best_tp_train = confusion_matrix(df_test['Class'], train_anom).ravel()
                best_pre_train = best_tn_train / (best_tp_train + best_fp_train) if (best_tp_train + best_fp_train) > 0 else 0
                best_rec_train = best_tp_train / (best_tp_train + best_fn_train) if (best_tp_train + best_fn_train) > 0 else 0
                best_f1_train = 2 * (best_pre_train * best_rec_train) / (best_pre_train + best_rec_train) if (best_pre_train + best_rec_train) > 0 else 0
            except ValueError:
                pass
            pd_data = {
                'name': filename,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'roc': roc_auc, 'pr': pr_auc,
                'best_tp': best_tp, 'best_tn': best_tn, 'best_fp': best_fp, 'best_fn': best_fn,
                'best_pre': best_pre, 'best_rec': best_rec, 'b_f_1': best_f1,
                'best_tp_train': best_tp_train, 'best_tn_train': best_tn_train, 'best_fp_train': best_fp_train, 'best_fn_train': best_fn_train,
                'best_pre_train': best_pre_train, 'best_rec_train': best_rec_train, 'b_f_1_train': best_f1_train
            }
            data_list.append(pd_data)
            # new_row = pd.DataFrame(pd_data, columns=columns_names, index=[0])
            # res_df = res_df._append(new_row, ignore_index=True)
            
            
            # pa_f1 = -1
            # for thr in thresholds:
            #     preds_pa = [True if s >= thr else False for s in scores]
            #     pa_prediction, t_latency, t_tp, t_tn, t_fp, t_fn, t_pre, t_rec, t_f1 = adjust_predicts(df_test['Class'], preds_pa, True)
            #     if t_f1 > pa_f1:
            #         latency, pa_tp, pa_tn, pa_fp, pa_fn, pa_pre, pa_rec, pa_f1 = t_latency, t_tp, t_tn, t_fp, t_fn, t_pre, t_rec, t_f1
                    
            # new_row1 = pd.Series([filename, pa_tp, pa_tn, pa_fp, pa_fn, pa_pre, pa_rec, pa_f1, latency],
            #                         index=['name', 'pa_tp', 'pa_tn', 'pa_fp', 'pa_fn', 'pa_pre', 'pa_rec', 'pa_f1', 'latency'])   
            # pa_df = pa_df._append(new_row1, ignore_index=True)
            
    res_df = pd.DataFrame(data_list, columns=columns_names, index=[0])
    res_df = add_summary_statistics(res_df)
    res_df.to_csv(f'results/{database}/{version}/{database}_results_woincon.csv')

    # pa_df = add_summary_statistics_pa(pa_df)
    # pa_df.to_csv('smd_5_results_pa.csv')