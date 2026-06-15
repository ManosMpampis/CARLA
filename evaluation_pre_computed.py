import numpy as np
import pandas as pd
import os
import warnings
from utils.utils import mkdir_if_missing
import argparse

warnings.filterwarnings("ignore", category=RuntimeWarning)
    
def add_summary_statistics(res_df):
    df_mean = res_df.mean(numeric_only=True).to_frame().transpose()
    df_mean["dataset"] = "mean"

    df_sum = res_df.sum(numeric_only=True).to_frame().transpose()
    df_sum["dataset"] = "sum"

    df_std = res_df.std(numeric_only=True).to_frame().transpose()
    df_std["dataset"] = "std"
    
    sum_tp = df_sum['tp'].item()
    sum_tn = df_sum['tn'].item()
    sum_fp = df_sum['fp'].item()
    sum_fn = df_sum['fn'].item()
    precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0
    recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    df_all = {
        "dataset": ["all_datasets"],
        "tp": [sum_tp],
        "tn": [sum_tn],
        "fp": [sum_fp],
        "fn": [sum_fn],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1_score],
        }
    res_df = pd.concat([res_df, df_sum, df_mean, df_std, pd.DataFrame.from_dict(df_all)], ignore_index=True)
    cols = ['dataset'] + [col for col in res_df.columns if col != 'dataset']
    res_df = res_df[cols]

    return res_df

def main(args):

    evaluation_types = [
        "eval_test_best",
        "eval_test_cls",
        "eval_test_train_th",
        "eval_timeseries_best",
        "eval_timeseries_cls",
        "eval_timeseries_train_th",
        "eval_train_best",
        "eval_train_cls"
        ]

    version = "layer_norm/small_eps_re_weight_initialization"#args.version
    save_dir = "_common"#args.save_dir
    ds_name = "new_smd"#args.get('dataset', 'new_smd')

    # mkdir_if_missing(save_dir)
    # save_dir = f'_{save_dir}' if (save_dir is not None and save_dir != "None") else ''
    
    folder = version #"big_train_custom_lr" #args.version

    data_info = os.listdir(f'results/{ds_name}/{folder}')
    files = [file for file in data_info if file.startswith('machine-')]
    files = sorted(files) if len(files) > 0 else [""]
    print(files)
    mkdir_if_missing(f'results/{ds_name}/{folder}/{save_dir}/best')
    mkdir_if_missing(f'results/{ds_name}/{folder}/{save_dir}/cls')
    df_best = {}
    df_cls = {}
    for et in evaluation_types:
        df_best_list = []
        df_cls_list = []
        for filename in files:
            if filename != 'machine-all.txt':
                temp_df_best = pd.read_csv(f"results/{ds_name}/{folder}/{filename}/classification{save_dir}/best/{et}.csv")
                temp_df_best['dataset'] = filename
                df_best_list.append(temp_df_best)

                temp_df_cls = pd.read_csv(f"results/{ds_name}/{folder}/{filename}/classification{save_dir}/best/{et}.csv")
                temp_df_cls['dataset'] = filename
                df_cls_list.append(temp_df_cls)
        df_best[et] = pd.concat(df_best_list, ignore_index=True)
        df_cls[et] = pd.concat(df_cls_list, ignore_index=True)
        
        df_best[et] = add_summary_statistics(df_best[et])
        df_cls[et] = add_summary_statistics(df_cls[et])

        df_best[et].to_csv(f'results/{ds_name}/{folder}/{save_dir}/best/results_{et}.csv')
        df_cls[et].to_csv(f'results/{ds_name}/{folder}/{save_dir}/cls/results_{et}.csv')


if __name__ == "__main__":
    FLAGS = argparse.ArgumentParser(description='classification Loss')
    FLAGS.add_argument('--version', help='Experiment version', type=str)
    FLAGS.add_argument('--save_dir', help='Save directory', type=str)
    args = FLAGS.parse_args()
    main(args)

