from torch import nn
import torch.nn.functional as F
import torch

from losses.soft_dtw_cuda import SoftDTW

EPS=1e-8


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
    

class EuclideanDistanceLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(EuclideanDistanceLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x1, x2):
        return torch.sum(((x1 - x2)**2), dim=-1) / self.temperature

def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))

def find_similarity_loss(loss_name, device, use_cuda=False, temperature=1.0):
    if loss_name == 'dtw':
        return DTWLoss(device, use_cuda=use_cuda, gamma=temperature)
    elif loss_name == 'euclidean':
        return EuclideanDistanceLoss(temperature=temperature)
    else:
        raise ValueError(f"Unsupported similarity loss: {loss_name}")