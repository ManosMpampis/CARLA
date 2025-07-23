import argparse
from carla_learner import CARLA

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='classification Loss')
    parser.add_argument('--config_env', help='Location of path config file', type=str, default='configs/env.yml')
    parser.add_argument('--config_exp', help='Location of experiments config file', type=str, default='configs/pretext/carla_pretext_smd.yml')
    parser.add_argument('--fname', help='Config the file name of Dataset', type=str, default='machine-1-1.txt')
    parser.add_argument('--device', help='Device used to load the model', type=str, choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--verbose', help='Enable logging messages level. 0: No verbose, 1: Terminal infor, 2: Terminal and file', type=int, choices=[0, 1, 2], default=2)
    parser.add_argument('--tensorboard', help='Enable tensorboard logging', action=argparse.BooleanOptionalAction, type=bool, default=True)
    args = parser.parse_args()

    version="2025-07-23-17-26-09"
    carla = CARLA(args.config_env, args.config_exp, args.fname, args.device, args.verbose, args.tensorboard, version=version)
    carla.train_pretext()

    