import os
import argparse

import numpy as np
import pandas as pd


def find_runs(labels):
    """Return list of (start, end) inclusive indices of consecutive 1s."""
    labels = np.asarray(labels).astype(int).ravel()
    padded = np.concatenate(([0], labels, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return list(zip(starts, ends))


def analyze_machine(label_path):
    labels = np.loadtxt(label_path).astype(int)
    runs = find_runs(labels)
    durations = np.array([e - s + 1 for s, e in runs], dtype=float)
    return {
        "machine": os.path.basename(label_path).replace(".txt", ""),
        "n_anomaly_events": len(runs),
        "n_events_under_256": int(np.sum(durations < 256)),
        "mean_window_duration": float(np.mean(durations)) if len(durations) else 0.0,
        "std_window_duration": float(np.std(durations)) if len(durations) else 0.0,
        "min_duration": float(np.min(durations)) if len(durations) else 0.0,
        "max_duration": float(np.max(durations)) if len(durations) else 0.0,
        "total_anomalous_points": int(labels.sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check SMD anomaly windows per machine sub-dataset."
    )
    parser.add_argument(
        "--root", default="datasets/SMD", help="SMD root folder (contains test_label/)"
    )
    parser.add_argument(
        "--label-dir", default="test_label", help="Subfolder with label files"
    )
    parser.add_argument("--out", default="smd_anomaly_windows.csv", help="Output csv")
    args = parser.parse_args()

    label_dir = os.path.join(args.root, args.label_dir)
    files = sorted(
        f
        for f in os.listdir(label_dir)
        if f.endswith(".txt") and "all" not in os.path.splitext(f)[0]
    )
    if not files:
        raise FileNotFoundError(f"No .txt label files found in {label_dir}")

    rows = [analyze_machine(os.path.join(label_dir, f)) for f in files]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved summary to {args.out}")


if __name__ == "__main__":
    main()
