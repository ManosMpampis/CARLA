import os
import glob
import pandas as pd
import argparse

metric_cols = [
    "train_calinski", "train_davies", "train_silhouette",
    "eval_calinski", "eval_davies", "eval_silhouette",
    "loss", "clear_loss",
]


def summarize(base_path, out_path, focused_tag=""):
    focused_tag = focused_tag if focused_tag=="" else f"_{focused_tag}"
    pattern = os.path.join(base_path, "**", f"pretext_evaluation{focused_tag}.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"No pretext_evaluation{focused_tag}.csv files found under {base_path}")
        return

    rows = []
    for fpath in sorted(files):
        rel = os.path.relpath(fpath, base_path)
        parts = rel.replace(".csv", "").split(os.sep)
        exp_label = parts[-2] if len(parts) > 1 else parts[0]
        if "batch" in parts:
            exp_label = f"batch_{exp_label}"
        elif "instance" in parts:
            exp_label = f"instance_{exp_label}"
        elif "layer" in parts:
            exp_label = f"layer_{exp_label}"
        elif "none" in parts:
            exp_label = f"none_{exp_label}"
        else:
            print("None normalization found")
        df = pd.read_csv(fpath)
        if "run" not in df.columns:
            continue
        mean_row = df[df["run"] == "mean"]
        if mean_row.empty:
            continue

        row = {"experiment": exp_label}
        for col in metric_cols:
            if col in mean_row.columns:
                row[col] = mean_row[col].iloc[0]
        rows.append(row)

    if not rows:
        print("No mean rows found in any CSV.")
        return

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print(f"Saved summary to {out_path} ({len(rows)} experiments)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, default="./results",
                        help="Root directory to search for pretext_evaluation.csv files")
    parser.add_argument("--out", type=str, default="./pretext_summary.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    
    focused_tag = [""] + metric_cols
    for tag in focused_tag:
        out = f"./pretext_summary{tag}.csv"
        summarize(args.base_path, out, focused_tag=tag)