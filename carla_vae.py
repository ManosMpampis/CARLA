"""Train a from-scratch Gaussian-bottleneck MSE+KL VAE comparison arm."""

import argparse

from utils.reconstruction_baselines import train_arm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_env", default="configs/env.yml")
    parser.add_argument("--config_exp", default="configs/baselines/smd_vae.yml")
    parser.add_argument("--fname", default="machine-1-1.txt")
    parser.add_argument("--version", default=None)
    train_arm("vae", parser.parse_args())
