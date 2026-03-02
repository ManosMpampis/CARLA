"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import torch.nn as nn

from losses.soft_dtw_cuda import SoftDTW


class DTWLoss(nn.Module):
    """Soft-DTW divergence loss function.
    https://arxiv.org/pdf/2010.08354.pdf
    """
    def __init__(self, device, use_cuda=False, gamma=.1):
        """Soft-DTW divergence loss function.

        Args:
            device (torch.device): Device.
            use_soft_dtw (bool, optional): Apply Soft-DTW loss function. Defaults to True.
            use_cuda (bool, optional): Apply Soft-DTW cuda implementation. Defaults to False.
            gamma (float, optional): DTW smoothing parameter. Defaults to .1.
        """
        super(DTWLoss, self).__init__()
        self.device = device
        self.soft_dtw = SoftDTW(use_cuda=use_cuda, gamma=gamma)

    def forward(self, vector_x, vector_y):
        """Forward pass.

        Args:
            vector_x (torch.Tensor): Batch tensor of dim (batch_size, win_size, n_feat).
            vector_y (torch.Tensor): Batch tensor of dim (batch_size, win_size, n_feat).

        Returns:
            torch.Tensor: Loss value.
        """
        loss = self.soft_dtw(vector_x, vector_y) \
        - .5 * (self.soft_dtw(vector_x, vector_x) + self.soft_dtw(vector_y, vector_y))
        return loss.to(self.device)
