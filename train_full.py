import argparse
from copy import deepcopy
import os
import subprocess

from carla_learner import CARLA

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_env', help='Location of path config file', type=str, default='configs/env.yml')
    parser.add_argument('--config_exp_pre', help='Location of experiments config file', type=str, default='configs/pretext/carla_pretext_smd.yml')
    parser.add_argument('--config_exp_class', help='Location of experiments config file', type=str, default='configs/classification/carla_classification_smd.yml')
    # parser.add_argument('--fname', help='Config the file name of Dataset', type=str, default='machine-1-1.txt')
    parser.add_argument('--device', help='Device used to load the model', type=str, choices=['cpu', 'cuda', 'auto'], default='auto')
    parser.add_argument('--verbose', help='Enable logging messages level. 0: No verbose, 1: Terminal infor, 2: Terminal and file', type=int, choices=[0, 1, 2], default=2)
    parser.add_argument('--tensorboard', help='Enable tensorboard logging', action=argparse.BooleanOptionalAction, type=bool, default=True)
    # parser.add_argument('--version', help='Add specific version name in the experiment', type=str, required=True)
    args = parser.parse_args()

    # version=args.version
    version='temp'
    # project_dir = os.path.dirname(__file__)
    # all_files = os.listdir(os.path.join(project_dir, 'datasets', 'SMD/train'))
    # file_list = [file for file in all_files if file.startswith('machine-')]
    # file_list = sorted(file_list)
    # print(file_list)
    
    # for idx, fname in enumerate(file_list):
    #     print(fname)
    #     if idx <= 10:
    #         continue
        
    #     subprocess.run([
    #         'python', 'train_pretext.py',
    #         '--config_env', 'configs/env.yml',
    #         '--config_exp', 'configs/pretext/carla_pretext_smd.yml',
    #         '--fname', fname,
    #         '--device', args.device,
    #         '--verbose', str(args.verbose),
    #         f'{"--tensorboard" if args.tensorboard else "--no-tensorboard"}',
    #         '--version', version
    #     ], check=True)
        
    #     # Run the classification script
    #     subprocess.run([
    #         'python', 'train_classification.py',
    #         '--config_env', 'configs/env.yml',
    #         '--config_exp', 'configs/classification/carla_classification_smd.yml',
    #         '--fname', fname,
    #         '--device', args.device,
    #         '--verbose', str(args.verbose),
    #         f'{"--tensorboard" if args.tensorboard else "--no-tensorboard"}',
    #         '--version', version
    #     ], check=True)
    fname = "machine-1-1.txt"  # Example file name, replace with your logic to select files
    # fname = "ALL"
    carla = CARLA(args.config_env, args.config_exp_pre, fname, args.device, args.verbose, args.tensorboard, version=version)
    version=deepcopy(carla.version)
    carla.train_pretext()
    carla.close()
    del carla
        
        # version="2025-07-23-19-07-25"
    # carla = CARLA(args.config_env, args.config_exp_class, fname, args.device, args.verbose, args.tensorboard, version=version)
    # carla.train_classification()
    # carla.close()
    # del carla
