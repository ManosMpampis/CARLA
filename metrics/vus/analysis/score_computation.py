
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
import random


import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from metrics.vus.utils.slidingWindows import find_length
from metrics.vus.utils.metrics import metricor

from metrics.vus.models.distance import Fourier
from metrics.vus.models.feature import Window

def find_section_length(label,length):
    best_i = None
    best_sum = None
    current_subseq = False
    for i in range(len(label)):
        changed = False
        if label[i] == 1:
            if current_subseq == False:
                current_subseq = True
                if best_i is None:
                    changed = True
                    best_i = i
                    best_sum = np.sum(label[max(0,i-200):min(len(label),i+9800)])
                else:
                    if np.sum(label[max(0,i-200):min(len(label),i+9800)]) < best_sum:
                        changed = True
                        best_i = i
                        best_sum = np.sum(label[max(0,i-200):min(len(label),i+9800)])
                    else:
                        changed = False
                if changed:
                    diff = i+9800 - len(label)

                    pos1 = max(0,i-200 - max(0,diff))
                    pos2 = min(i+9800,len(label))
        else:
            current_subseq = False
    if best_i is not None:
        return best_i-pos1,(pos1,pos2)
    else:
        return None,None

def generate_data(filepath,init_pos,max_length):
    
    df = pd.read_csv(filepath, header=None).to_numpy()
    name = filepath.split('/')[-1]
    #max_length = 30000
    data = df[init_pos:init_pos+max_length,0].astype(float)
    label = df[init_pos:init_pos+max_length,1]
    
    pos_first_anom,pos = find_section_length(label,max_length)
    
    data = df[pos[0]:pos[1],0].astype(float)
    label = df[pos[0]:pos[1],1]
    
    slidingWindow = find_length(data)
    #slidingWindow = 70
    X_data = Window(window = slidingWindow).convert(data).to_numpy()

    data_train = data[:int(0.1*len(data))]
    data_test = data

    X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
    X_test = Window(window = slidingWindow).convert(data_test).to_numpy()
    
    return pos_first_anom,slidingWindow,data,X_data,data_train,data_test,X_train,X_test,label





