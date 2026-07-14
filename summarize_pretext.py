import os
import glob
import pandas as pd
import argparse

metric_cols = [
    "train_calinski", "train_davies", "train_silhouette",
    "eval_calinski", "eval_davies", "eval_silhouette",
    "loss", "clear_loss", "neg_included", "neg_not_included",
    "pos_included", "pos_not_included", "margin",
    "negative_d_loss", "positive_d_loss",
]


def summarize(base_path, out_path, focused_tag=""):
    pattern = os.path.join(base_path, "**", f"pretext_evaluation{focused_tag}.csv")
    files = glob.glob(pattern, recursive=True)

    if not files:
        print(f"No pretext_evaluation{focused_tag}.csv files found under {base_path}")
        return

    rows = []
    for fpath in sorted(files):
        rel = os.path.relpath(fpath, base_path)
        parts = rel.replace(".csv", "").split(os.sep)
        exp_label = " / ".join(parts[:-1]) if len(parts) > 1 else parts[0]

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

    summarize(args.base_path, args.out)
