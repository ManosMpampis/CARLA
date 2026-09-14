#!/usr/bin/env python3
"""Aggregate per-machine metrics.json files into per-experiment and summary CSVs.

Layout expected:
    <base>/<experiment>/machine*/jepa/metrics.json
e.g.:
    results/smd/lewm/configs/jepa/lewm/smd_lewm_full128_pretrain.yml/machine-1-1.txt/jepa/metrics.json

Outputs (under --outdir):
    <outdir>/<experiment>.csv   one row per machine + MEAN/STD/SUM rows
    <outdir>/summary_mean.csv   one row per experiment (mean across machines)

Usage:
    ./venv/bin/python aggregate_lewm_metrics.py
    ./venv/bin/python aggregate_lewm_metrics.py --base results/smd/lewm/configs/jepa/lewm --outdir results/smd/lewm/aggregated_metrics
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def flatten(d: dict, prefix: str = "", sep: str = "/") -> dict:
    """Flatten nested dicts: {"honest": {"point_AUROC": x}} -> {"honest/point_AUROC": x}."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key, sep))
        else:
            out[key] = v
    return out


def aggregate_values(rows: list[dict], columns: list[str]):
    """Return (means, stds, sums) dicts over numeric values per column.

    STD is the sample standard deviation (ddof=1); 0.0 when n < 2.
    Non-numeric / missing entries are skipped.
    """
    means, stds, sums = {}, {}, {}
    for col in columns:
        if "Affiliation precision" in col:
            continue
        vals = [r[col] for r in rows if isinstance(r.get(col), (int, float))]
        if not vals:
            means[col] = stds[col] = sums[col] = ""
            continue
        sums[col] = float(sum(vals))
        means[col] = float(sum(vals) / len(vals))
        stds[col] = float(statistics.stdev(vals)) if len(vals) >= 2 else 0.0
    return means, stds, sums


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="results/smd/frequency/configs/jepa/steering",
                    help="Directory containing one subdir per experiment")
    ap.add_argument("--outdir", default="results/smd/frequency/aggregated_metrics",
                    help="Where to write per-experiment CSVs + summary_mean.csv")
    args = ap.parse_args()

    base = Path(args.base)
    # Tolerate the common lewn/lewm typo: fall back to the existing sibling.
    if not base.is_dir() and "lewn" in str(base):
        alt = Path(str(base).replace("lewn", "lewm"))
        if alt.is_dir():
            print(f"[warn] {base} not found, using {alt}", file=sys.stderr)
            base = alt
    if not base.is_dir():
        print(f"[error] base dir not found: {base}", file=sys.stderr)
        return 1

    experiments = sorted(p for p in base.iterdir() if p.is_dir())
    if not experiments:
        print(f"[error] no experiment subdirs in {base}", file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    summary_columns: list[str] = []

    for exp in experiments:
        files = sorted(exp.glob("machine*/jepa/metrics.json"))
        if not files:
            print(f"[warn] {exp.name}: no machine*/jepa/metrics.json, skipped", file=sys.stderr)
            continue
        rows: list[dict] = []
        columns: list[str] = []
        for f in files:
            machine = f.parent.parent.name  # machine-*.txt
            with open(f) as fh:
                flat = flatten(json.load(fh))
            for col in flat:
                if col not in columns:
                    columns.append(col)
            rows.append({"machine": machine, **flat})
        columns.sort()
        means, stds, sums = aggregate_values(rows, columns)

        # Per-experiment CSV: one row per machine + MEAN/STD/SUM.
        per_exp_path = outdir / f"{exp.name}.csv"
        with open(per_exp_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["machine", *columns])
            w.writeheader()
            for r in sorted(rows, key=lambda r: r["machine"]):
                w.writerow({k: r.get(k, "") for k in ["machine", *columns]})
            w.writerow({"machine": "MEAN", **means})
            w.writerow({"machine": "STD", **stds})
            w.writerow({"machine": "SUM", **sums})
        print(f"[ok] {exp.name}: {len(rows)} machines -> {per_exp_path}")

        for col in columns:
            if col not in summary_columns:
                summary_columns.append(col)
        summary_rows.append({"experiment": exp.name, **means})

    summary_columns.sort()
    summary_path = outdir / "summary_mean.csv"
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["experiment", *summary_columns])
        w.writeheader()
        for r in sorted(summary_rows, key=lambda r: r["experiment"]):
            w.writerow({k: r.get(k, "") for k in ["experiment", *summary_columns]})
    print(f"[ok] summary ({len(summary_rows)} experiments) -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
