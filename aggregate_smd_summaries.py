#!/usr/bin/env python3
"""Compare experiments across machine*/summary.csv files (equal machine weights).

Usage: ./venv/bin/python aggregate_smd_summaries.py
STD uses ddof=1; a single valid value has STD 0. Missing/nonfinite values
and rows whose status is not 'completed' are excluded from metric statistics.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('results/smd/summaries'))
    parser.add_argument('--output', type=Path, default=Path('smd_summary_aggregated.csv'))
    parser.add_argument('--metrics', nargs='+', default=[
        'timeseries_f1', 'window_f1', 'vus_pr', 'vus_roc',
    ], help='Numeric summary columns to aggregate')
    args = parser.parse_args()
    files = sorted(args.base.glob('machine*/summary.csv'))
    if not files:
        parser.error(f'No machine*/summary.csv files found under {args.base}')

    groups = {}
    for path in files:
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'experiment_key', 'run', 'status', *args.metrics}
            missing = required - set(reader.fieldnames or [])
            if missing:
                parser.error(f'{path}: missing columns: {", ".join(sorted(missing))}')
            for row in reader:
                key = (row['experiment_key'], row['run'])
                machines = groups.setdefault(key, {})
                machine = path.parent.name
                if machine in machines:
                    parser.error(f'{path}: duplicate experiment/run: {key}')
                machines[machine] = row

    columns = ['experiment_key', 'run', 'machines_total', 'machines_present',
               'machines_completed']
    for metric in args.metrics:
        columns.extend([f'{metric}_count', f'{metric}_mean', f'{metric}_std'])
    results = []
    for (experiment, run), machines in sorted(groups.items()):
        completed = [row for row in machines.values() if row['status'] == 'completed']
        result = dict(experiment_key=experiment, run=run, machines_total=len(files),
                      machines_present=len(machines), machines_completed=len(completed))
        for metric in args.metrics:
            values = []
            for row in completed:
                try:
                    value = float(row[metric])
                except (ValueError, TypeError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            result[f'{metric}_count'] = len(values)
            result[f'{metric}_mean'] = statistics.mean(values) if values else ''
            result[f'{metric}_std'] = (
                statistics.stdev(values) if len(values) > 1 else 0.0 if values else ''
            )
        results.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    print(f'Aggregated {len(files)} machines, {len(results)} experiment/run groups -> {args.output}')


if __name__ == '__main__':
    main()
