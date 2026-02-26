"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import numpy as np
import torch
import torch.nn as nn
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from losses.soft_dtw_cuda import SoftDTW


class DTWLoss(nn.Module):
    """Soft-DTW divergence loss function.
    https://arxiv.org/pdf/2010.08354.pdf
    """
    def __init__(self, device, use_soft_dtw=True,
                use_cuda=False, gamma=.1):
        """Soft-DTW divergence loss function.

        Args:
            device (torch.device): Device.
            use_soft_dtw (bool, optional): Apply Soft-DTW loss function. Defaults to True.
            use_cuda (bool, optional): Apply Soft-DTW cuda implementation. Defaults to False.
            gamma (float, optional): DTW smoothing parameter. Defaults to .1.
        """
        super(DTWLoss, self).__init__()
        self.device = device
        self.use_soft_dtw = use_soft_dtw
        self.soft_dtw = SoftDTW(use_cuda=use_cuda, gamma=gamma)

    def forward(self, vector_x, vector_y):
        """Forward pass.

        Args:
            vector_x (torch.Tensor): Batch tensor of dim (batch_size, win_size, n_feat).
            vector_y (torch.Tensor): Batch tensor of dim (batch_size, win_size, n_feat).

        Returns:
            torch.Tensor: Loss value.
        """
        # vector_x = vector_x.to("cpu")
        # vector_y = vector_y.to("cpu")
        if self.use_soft_dtw:
            loss = self.soft_dtw(vector_x, vector_y) \
            - .5 * (self.soft_dtw(vector_x, vector_x) + self.soft_dtw(vector_y, vector_y))
        else:
            dist_list = [fastdtw(vector_x[i].detach().numpy(),
                                vector_y[i].detach().numpy(),
                                dist=euclidean)
                            for i in range(vector_x.size(0))
                            ]
            dist_list = [dist_tup[0] for dist_tup in dist_list]
            dist_array = np.array(dist_list)
            loss = torch.from_numpy(dist_array)
        #loss = torch.mean(loss)
        # vector_x = vector_x.to(self.device)
        # vector_y = vector_y.to(self.device)
        return loss.to(self.device)
