from tbparse import SummaryReader
from torch import load
import os
import re
import glob
import numpy as np
import pandas as pd

logs_check_tags = {
    "train_calinski": "Pretext Evaluation/Calinski-Harabasz Score",
    "train_davies": "Pretext Evaluation/Davies-Bouldin Score",
    "train_silhouette": "Pretext Evaluation/Silhouette Score",
    "eval_calinski": "Pretext Evaluation_eval/Calinski-Harabasz Score",
    "eval_davies": "Pretext Evaluation_eval/Davies-Bouldin Score",
    "eval_silhouette": "Pretext Evaluation_eval/Silhouette Score",
    "loss": "Pretext Loss/loss",
    "clear_loss": "Pretext Loss/clear_loss",
}

metric_tags = {
    "train_calinski": "Pretext Evaluation/Calinski-Harabasz Score",
    "train_davies": "Pretext Evaluation/Davies-Bouldin Score",
    "train_silhouette": "Pretext Evaluation/Silhouette Score",
    "eval_calinski": "Pretext Evaluation_eval/Calinski-Harabasz Score",
    "eval_davies": "Pretext Evaluation_eval/Davies-Bouldin Score",
    "eval_silhouette": "Pretext Evaluation_eval/Silhouette Score",
    "loss": "Pretext Loss/loss",
    "clear_loss": "Pretext Loss/clear_loss",
    "neg_included": "Pretext Loss/loss_neg_c",
    "neg_not_included": "Pretext Loss/loss_neg_nc",
    "pos_included": "Pretext Loss/loss_pos_c",
    "pos_not_included": "Pretext Loss/loss_pos_nc",
    "margin": "Pretext Loss/margin",
    "negative_d_loss": "Pretext Loss/negative_d_loss",
    "positive_d_loss": "Pretext Loss/positive_d_loss",
}
metric_cols = list(metric_tags.keys())


def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _find_path(root, name):
    matches = glob.glob(os.path.join(root, "**", name), recursive=True)
    return matches[0] if matches else None


def _metric_value(df, tag, best_epoch):
    sub = df[df["tag"] == tag]
    if sub.empty:
        return np.nan
    dist = (sub["step"] - best_epoch).abs()
    closest = sub[dist == dist.min()]
    return float(closest["value"].values[-1])

def _best_epoch_metric_value(df, tag, min=False):
    sub = df[df["tag"] == tag]
    if sub.empty:
        return np.nan
    max_tag_value = sub["value"].max() if not min else sub["value"].min()
    sub = sub[sub["value"] == max_tag_value]
    step = (sub["step"].iloc[-1])
    return step

def evaluate(experiment_cluster, ds_name="smd", big_exp_name="", base_path=None):
    if base_path is None:
        base_path = f"./results/{ds_name}/{big_exp_name}"

    for exp_cluster in experiment_cluster:
        exp_cluster_dir = os.path.join(base_path, exp_cluster)
        experiments = [d for d in os.listdir(exp_cluster_dir) if "machine" not in d]
        if not experiments:
            experiments = ["/"]
        for exp in experiments:
            # if not exp == "original":
            #     continue
            exp_dir = os.path.join(exp_cluster_dir, exp)
            if not os.path.isdir(exp_dir):
                print(f"[skip] not found: {exp_dir}")
                continue

            run_dirs = [d for d in os.listdir(exp_dir) if d.startswith("machine-")]
            run_dirs = sorted(run_dirs, key=_natural_key)

            rows = []
            rows_logs_check_tags = {key: [] for key in logs_check_tags.keys()}
            for run in run_dirs:
                run_path = os.path.join(exp_dir, run)
                ckpt = _find_path(run_path, "checkpoint.pth_best.pth.tar")
                if ckpt is None:
                    print(f"[skip] no checkpoint: {run_path}")
                    continue
                ckpt_dir = os.path.dirname(ckpt)
                tb_dir = os.path.join(ckpt_dir, "tensorboard")
                if not os.path.isdir(tb_dir):
                    tb_matches = glob.glob(
                        os.path.join(run_path, "**", "tensorboard"), recursive=True
                    )
                    tb_dir = tb_matches[0] if tb_matches else None
                if tb_dir is None:
                    print(f"[skip] no tensorboard: {run_path}")
                    continue

                save_info = load(ckpt, map_location="cpu", weights_only=False)
                best_epoch = save_info["epoch"]

                df = SummaryReader(tb_dir, event_types=set(["scalars"])).scalars

                label = run[:-4] if run.endswith(".txt") else run
                row = {f"run": label, "epoch": best_epoch}
                for mname, tag in metric_tags.items():
                    row[mname] = _metric_value(df, tag, best_epoch)
                for values, base_tag in logs_check_tags.items():
                    epoch_tag = _best_epoch_metric_value(df, base_tag, min= ("Davies-Bouldin Score" in base_tag))
                    tag_row = {f"run": label, "epoch": epoch_tag}
                    for mname, tag in metric_tags.items():
                        tag_row[mname] = _metric_value(df, tag, epoch_tag)
                    rows_logs_check_tags[values].append(tag_row)
                rows.append(row)

            if not rows:
                print(f"[skip] no runs processed for {exp}")
                continue

            df_out = pd.DataFrame(rows)
            agg = pd.DataFrame({"run": ["mean", "std", "sum"]})
            for col in metric_cols:
                agg[col] = [df_out[col].mean(), df_out[col].std(), df_out[col].sum()]
            df_out = pd.concat([df_out, agg], ignore_index=True)

            out_path = os.path.join(exp_dir, "pretext_evaluation.csv")
            df_out.to_csv(out_path, index=False)
            print(f"Saved {out_path} ({len(rows)} runs)")

            for values, rows in rows_logs_check_tags.items():
                df_out = pd.DataFrame(rows)
                agg = pd.DataFrame({"run": ["mean", "std", "sum"]})
                for col in metric_cols:
                    agg[col] = [df_out[col].mean(), df_out[col].std(), df_out[col].sum()]
                df_out = pd.concat([df_out, agg], ignore_index=True)

                out_path = os.path.join(exp_dir, f"pretext_evaluation_{values}.csv")
                df_out.to_csv(out_path, index=False)
                print(f"Saved {out_path} ({len(rows)} runs)")


if __name__ == "__main__":
    experiment_cluster = ["instance", "layer", "none"] #["original-dynamic_weight-loss_clamp", "ema_loss", "dynamic_reweight_on_distance"]
    ds_name = "smd"
    big_exp_name = "normalization-strategy" #"batch"
    evaluate(experiment_cluster, ds_name, big_exp_name)
