"""Run every configured PSM framework and named variant; print/save results."""

from pathlib import Path

from utils.suite_cli import suite_cli


def cli():
    suite_cli(Path(__file__).resolve().parent / "configs/psm/experiments.yml", "psm", __doc__)


if __name__ == "__main__":
    cli()
