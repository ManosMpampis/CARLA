"""Train or score a Gaussian-bottleneck MSE+KL VAE comparison arm."""

import argparse

from utils.reconstruction_baselines import run_arm


def main(args, update_dictionary=None):
    return run_arm("vae", args, update_dictionary)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_env", default="configs/env.yml")
    parser.add_argument("--config_exp", default="configs/baselines/smd_vae.yml")
    parser.add_argument("--fname", default="machine-1-1.txt")
    parser.add_argument("--version", default=None)
    parser.add_argument("--score", action="store_true", help="score saved validation-selected weights using this config")
    parser.add_argument("--score_checkpoint", help="optional weights path")
    main(parser.parse_args())


if __name__ == "__main__":
    cli()
