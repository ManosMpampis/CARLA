
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from termcolor import colored
from utils.config import create_config
from utils.common_config import get_model

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLAGS = argparse.ArgumentParser(description='classification Loss')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--fname', help='Config the file name of Dataset')
FLAGS.add_argument('--version', help='Experiment version', type=str)

class Carla():
    def __init__(self, config_env, config_exp, fname, version):
        self.p = create_config(config_env, config_exp, fname, version)
        self.model = get_model(self.p, self.p['pretext_model'])

        print(colored('\n- Model initialisation', 'green'))
        
        # Checkpoint
        model_path = self.p['classification_model'] if os.path.exists(self.p['classification_model']) else self.p['classification_checkpoint']
        if os.path.exists(model_path):
            print(colored('-- Model initialised from selected model: {}'.format(model_path), 'green'))
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            self.normal_label = checkpoint['normal_label']
        else:
            self.normal_label = 0
        
        self.model.to(device)
        self.model.eval()

    def predict(self, ts):
        if ts.ndim == 3:
            bs, w, h = ts.shape
        else:
            bs, w = ts.shape
            h =1
        
        res = self.model(ts.reshape(bs, h, w), forward_pass='return_all')
        output = res['output']
        predictions = torch.argmax(output, dim=1).cpu().numpy()
        anomalies = np.where(predictions == self.normal_label, 0, 1)
        return anomalies

def main():
    args = FLAGS.parse_args()
    classifer = Carla(args.config_env, args.config_exp, args.fname, args.version)
    
    ts = ...
    anomalies = classifer.predict(ts)


if __name__ == "__main__":
    main()
