"""Train a from-scratch deterministic MSE autoencoder comparison arm."""

import argparse

from utils.reconstruction_baselines import train_arm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_env", default="configs/env.yml")
    parser.add_argument("--config_exp", default="configs/baselines/smd_ae.yml")
    parser.add_argument("--fname", default="machine-1-1.txt")
    parser.add_argument("--version", default=None)
    train_arm("ae", parser.parse_args())
