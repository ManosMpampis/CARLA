"""Alias for the unified experiment CLI; use --dataset tsb-ad-u or tsb-ad-m.

The removed legacy benchmark orchestration is replaced by train.py's
single-series/CSV-list selection, preflight, native trainer and bounded jobs.
Legacy benchmark-specific flags are not supported.
"""

from train import cli


if __name__ == "__main__":
    raise SystemExit(cli())
