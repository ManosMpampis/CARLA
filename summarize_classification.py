import os
import glob
import argparse
import pandas as pd


def summarize(agg_path, out_path):
    pattern = os.path.join(agg_path, "classification_*", "*", "*.csv")
    files = [
        f for f in glob.glob(pattern)
        if os.path.basename(os.path.dirname(f)) in ("best", "cls")
    ]

    if not files:
        print(f"No aggregated CSVs found under {agg_path}. "
              "Run aggregate_classification.py first.")
        return

    # group files by (split, csv_name)
    groups = {}
    for fpath in files:
        split = os.path.basename(os.path.dirname(fpath))  # best | cls
        experiment = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
        csv_name = os.path.basename(fpath)
        groups.setdefault((split, csv_name), []).append((experiment, fpath))

    for (split, csv_name), entries in sorted(groups.items()):
        rows = []
        for experiment, fpath in sorted(entries):
            df = pd.read_csv(fpath)
            mean_row = df[df["machine"] == "mean"]
            if mean_row.empty:
                continue
            row = {"experiment": experiment}
            row.update(mean_row.iloc[0].drop(labels=["machine"]).to_dict())
            rows.append(row)

        if not rows:
            continue

        summary = pd.DataFrame(rows)
        out_dir = os.path.join(out_path, split)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, csv_name)
        summary.to_csv(out_file, index=False)
        print(f"Saved {out_file} ({len(rows)} experiments)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build one summary CSV per split (best/cls) and csv type, "
                    "taking only the mean row of each classification experiment "
                    "from the aggregated results."
    )
    parser.add_argument("--agg-path", type=str,
                        default="./results/smd/best_models/batch/original/aggregated",
                        help="Root directory with the output of aggregate_classification.py")
    parser.add_argument("--out", type=str,
                        default="./results/smd/best_models/batch/original/aggregated/summary",
                        help="Output root directory for the summary CSVs")
    args = parser.parse_args()

    summarize(args.agg_path, args.out)
