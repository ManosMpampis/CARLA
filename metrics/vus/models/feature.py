# -*- coding: utf-8 -*-
"""Classes of feature mapping for model type B
"""

import numpy as np
# import matplotlib.pyplot as plt
# import random
# from arch import arch_model
import pandas as pd
import math
# import pmdarima as pm
# from pmdarima import model_selection
# import os
# import dis
# import statistics
# from sklearn import metrics
# import sklearn
# from tsfresh import extract_features

from statsmodels.tsa.seasonal import seasonal_decompose

# import itertools
# import functools
import warnings
from builtins import range
# from collections import defaultdict


from numpy.linalg import LinAlgError
# from scipy.signal import cwt, find_peaks_cwt, ricker, welch
# from scipy.stats import linregress
# from statsmodels.tools.sm_exceptions import MissingDataError

with warnings.catch_warnings():
    # Ignore warnings of the patsy package
    warnings.simplefilter("ignore", DeprecationWarning)

    from statsmodels.tsa.ar_model import AR
# from statsmodels.tsa.stattools import acf, adfuller, pacf

# from hurst import compute_Hc

class Window:
    """ The  class for rolling window feature mapping.
    The mapping converts the original timeseries X into a matrix. 
    The matrix consists of rows of sliding windows of original X. 
    """

    def __init__(self,  window = 100):
        self.window = window
        self.detector = None
    def convert(self, X):
        n = self.window
        X = pd.Series(X)
        L = []
        if n == 0:
            df = X
        else:
            for i in range(n):
                L.append(X.shift(i))
            df = pd.concat(L, axis = 1)
            df = df.iloc[n-1:]
        return df
