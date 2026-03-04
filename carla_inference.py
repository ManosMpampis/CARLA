
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
        if os.path.exists(self.p['classification_checkpoint']):
            print(colored('-- Model initialised from last checkpoint: {}'.format(self.p['classification_checkpoint']), 'green'))
            checkpoint = torch.load(self.p['classification_checkpoint'], map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            self.normal_label = checkpoint['normal_label']
        else:
            self.normal_label = 0
        
        self.model.to(device)
        self.model.eval()

    def predict(self, ts):
        probs = []
        predictions = []

        if ts.ndim == 3:
            bs, w, h = ts.shape
        else:
            bs, w = ts.shape
            h =1
        
        res = self.model(ts.view(bs, h, w), forward_pass='return_all')
        output = res['output']
        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1))
            probs[i].append(F.softmax(output_i, dim=1))

        anomalies = np.where((predictions == self.normal_label), 0, 1)
        scores = 1-np.array(probs)[:, self.normal_label]
        return anomalies

def main():
    args = FLAGS.parse_args()
    classifer = Carla(args.config_env, args.config_exp, args.fname, args.version)
    
    ts = ...
    anomalies = classifer.predict(ts)


if __name__ == "__main__":
    main()
