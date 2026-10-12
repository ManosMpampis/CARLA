"""Run every configured experiment on SMD, with separate runs per machine."""

from pathlib import Path

from utils.suite_cli import suite_cli


def cli():
    suite_cli(Path(__file__).resolve().parent / "configs/smd/experiments.yml", "smd", __doc__)


if __name__ == "__main__":
    cli()
