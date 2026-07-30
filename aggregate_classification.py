import os
import re
import glob
import argparse
import pandas as pd


def natural_key(name):
    """Natural sort key so machine-1-2 sorts before machine-1-10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def aggregate(base_path, out_path):
    pattern = os.path.join(base_path, "machine-*", "classification_*", "*", "*.csv")
    files = [
        f for f in glob.glob(pattern)
        if os.path.basename(os.path.dirname(f)) in ("best", "cls")
    ]

    if not files:
        print(f"No classification result CSVs found under {base_path}")
        return

    # group files by (experiment, split, csv_name)
    groups = {}
    for fpath in files:
        split = os.path.basename(os.path.dirname(fpath))  # best | cls
        experiment = os.path.basename(os.path.dirname(os.path.dirname(fpath)))
        machine = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(fpath))))
        machine = machine.replace(".txt", "")
        csv_name = os.path.basename(fpath)
        groups.setdefault((experiment, split, csv_name), []).append((machine, fpath))

    for (experiment, split, csv_name), entries in sorted(groups.items()):
        rows = []
        for machine, fpath in sorted(entries, key=lambda x: natural_key(x[0])):
            df = pd.read_csv(fpath)
            if df.empty:
                continue
            row = {"machine": machine}
            row.update(df.iloc[0].to_dict())
            rows.append(row)

        if not rows:
            continue

        combined = pd.DataFrame(rows)
        metric_cols = [c for c in combined.columns if c != "machine"]
        values = combined[metric_cols].apply(pd.to_numeric, errors="coerce")

        stats = pd.DataFrame([values.mean(), values.std(), values.sum(min_count=1)])
        stats.insert(0, "machine", ["mean", "std", "sum"])
        combined = pd.concat([combined, stats], ignore_index=True)

        out_dir = os.path.join(out_path, experiment, split)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, csv_name)
        combined.to_csv(out_file, index=False)
        print(f"Saved {out_file} ({len(rows)} machines + mean/std/sum)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine per-machine classification results into one CSV per "
                    "classification experiment, split (best/cls) and csv type, "
                    "with mean/std/sum rows appended."
    )
    parser.add_argument("--base-path", type=str,
                        default="./results/smd/best_models/batch/original",
                        help="Root directory containing the machine-* folders")
    parser.add_argument("--out", type=str,
                        default="./results/smd/best_models/batch/original/aggregated",
                        help="Output root directory for the aggregated CSVs")
    args = parser.parse_args()

    aggregate(args.base_path, args.out)
