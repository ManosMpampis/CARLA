"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import numpy as np
import torch
import torch.nn as nn

from losses.dtw_loss import DTWLoss

class TCLoss(nn.Module):
    """Temporal Contrastive Loss.
    """
    def __init__(self, bs, device, gamma=1, crop_size_min=5, crop_size_max=10,
                use_cuda=False, max_margin=5, min_margin=1, temperature=.1, margin=5):
        """_summary_

        Args:
            loss_fn (torch.Module): Similarity loss.
            device (torch.device): Device.
            crop_size_min (int, optional): Min crop size. Defaults to 5.
            crop_size_max (int, optional): Max crop size. Defaults to 10.
            if_use_dtw (bool, optional): Apply DTW similarity. Defaults to False.
            max_margin (int, optional): Max margin value. Defaults to 5.
            min_margin (int, optional): Min margin value. Defaults to 1.
            num_clusters (int, optional): Number of clusters. Defaults to 2.
            temperature (float, optional): Temperature parameter. Defaults to .1.
            margin (int, optional): Margin value. Defaults to 5.
        """
        super(TCLoss, self).__init__()
        self.sim_loss = DTWLoss(device, use_cuda=use_cuda, gamma=gamma)
        
        self.bs = bs

        self.crop_size_min = crop_size_min
        self.crop_size_max = crop_size_max
        self.max_margin = max_margin
        self.min_margin = min_margin
        self.margin = margin
        self.device = device
        self.temperature = temperature

    def update_margin(self,  new_margin):
        if new_margin is None:
            return
        
        self.margin = new_margin.to(self.device) if isinstance(new_margin, torch.Tensor) else torch.tensor(new_margin).to(self.device)

    def random_crop(self, data1, data3=None):
        """Apply random cropping.

        Args:
            data1 (torch.Tensor): Batch of temporal views.
            data3 (torch.Tensor, optional): Batch of negative views. Defaults to None.

        Returns:
            (torch.Tensor): Triplet of temporal views.
        """
        crop_size = np.random.randint(self.crop_size_min, self.crop_size_max)

        max_start_index = data1.size(1) - crop_size + 1
        start_index = np.random.randint(0, max_start_index)
        crop_data_1 = data1[:, start_index : start_index + crop_size, :]

        start_index = np.random.randint(0, max_start_index)
        crop_data_2 = data1[:, start_index : start_index + crop_size, :]


        if data3 is not None:
            crop_data_3 = []
            for data in data3:
                start_index = np.random.randint(0, max_start_index)
                crop = data[:, start_index : start_index + crop_size, :]

                crop_data_3.append(crop)
        else:
            crop_data_3 = None
        return crop_data_1, crop_data_2, crop_data_3

    def temporal_triplet_loss(self, crop_z1, crop_z2, crop_z3=None, update=False):
        """Compute triplet loss using triplet of views.

        Args:
            crop_z1 (torch.Tensor): Positive crop views.
            crop_z2 (torch.Tensor): Positive crop views.
            crop_z3 (torch.Tensor, optional): Negative crop views. Defaults to None.
            update (bool, optional): Update margin. Defaults to False.

        Returns:
            torch.Tensor: Loss tensor.
        """
        # Compute the temporal loss using soft-DTW-triplet loss
        loss_pos = self.sim_loss(crop_z1, crop_z2) # positive distance

        if crop_z3 is not None:
            loss_neg = self.sim_loss(crop_z1, crop_z3)

            # Init margin
            if self.margin is None:
                self.margin = self.max_margin

            if update:
                # update margin
                dist_pos_neg = self.sim_loss(crop_z2, crop_z3)
                new_margin = torch.clamp(self.max_margin - torch.mean(dist_pos_neg),
                                        min=self.min_margin)
                self.update_margin(new_margin.item())

            loss = torch.clamp(loss_pos - loss_neg + self.margin, min=0.0)


        loss_m = torch.mean(loss)
        loss_m_pos = torch.mean(loss_pos)
        loss_m_neg = torch.mean(loss_neg) if crop_z3 is not None else 0

        mask = loss > 0
        loss_pos_c = loss_pos[mask]
        loss_neg_c = loss_neg[mask] if crop_z3 is not None else 0
        mask = loss <= 0
        loss_pos_nc = loss_pos[mask]
        loss_neg_nc = loss_neg[mask] if crop_z3 is not None else 0
        return {"loss": loss_m, "positive_d_loss": loss_m_pos, "negative_d_loss": loss_m_neg, "positive_d_loss_c": torch.mean(loss_pos_c), "negative_d_loss_c": torch.mean(loss_neg_c), "positive_d_loss_nc": torch.mean(loss_pos_nc), "negative_d_loss_nc": torch.mean(loss_neg_nc)}


    def forward(self,
                features,
                update=False,
                crop=False):
        """Forward pass.

        Args:
            z1_batch (torch.Tensor): Batch of positive views.
            z2_batch (torch.Tensor): Batch of 2nd positive views.
            z3_batch (torch.Tensor, optional): Batch of negative views. Defaults to None.
            update (bool, optional): Update margin. Defaults to False.
            crop (bool, optional): Apply random cropping. Defaults to True.
            
        Returns:
            torch.Tensor: Loss tensor.
        """

        z1_batch, z2_batch, z3_batch = torch.split(features, self.bs, dim=0)
        z1_batch = z1_batch.view(self.bs, -1, z1_batch.shape[1])
        z2_batch = z2_batch.view(self.bs, -1, z2_batch.shape[1])
        z3_batch = z3_batch.view(self.bs, -1, z3_batch.shape[1])
        if crop:
            crop_z1, crop_z2, crop_z3 = self.random_crop(z1_batch, z3_batch)
            losses = self.temporal_triplet_loss(crop_z1, crop_z2, crop_z3, update)
            
            crop_z1, crop_z2, crop_z3 = self.random_crop(z2_batch, z3_batch)
            losses2 = self.temporal_triplet_loss(crop_z1, crop_z2, crop_z3, update)
            
            for loss in losses.keys():
                losses[loss] = (losses[loss] + losses2[loss]) / 2
        else:
            losses = self.temporal_triplet_loss(z1_batch, z2_batch, z3_batch)

        return losses
