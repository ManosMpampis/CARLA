from torch import nn
import torch.nn.functional as F
import torch
from typing import cast

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

class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (sliced Epps-Pulley).

    Latent tokens are projected onto a fixed set of random unit slices;
    per slice the empirical characteristic function is compared with the
    standard-normal one over a quadrature grid of frequencies. A constant
    (collapsed) latent maximizes the statistic, an isotropic Gaussian
    minimizes it. Projections are deterministic given the latent dim, so
    runs are reproducible and the buffer travels inside checkpoints.
    """

    def __init__(self, num_slices: int = 16, freq_nodes: int = 8,
                 freq_min: float = 0.2, freq_max: float = 4.0, seed: int = 4):
        super().__init__()
        self.num_slices = num_slices
        self.register_buffer(
            "freqs",
            torch.linspace(freq_min, freq_max, freq_nodes),
            persistent=False,
        )
        self._seed = seed
        self._slices: dict[int, torch.Tensor] = {}

    def _get_slices(self, dim: int, device, dtype) -> torch.Tensor:
        if dim not in self._slices:
            generator = torch.Generator().manual_seed(self._seed + dim)
            directions = torch.randn(self.num_slices, dim, generator=generator)
            directions = directions / directions.norm(dim=1, keepdim=True)
            self._slices[dim] = directions.to(device=device, dtype=dtype)
        return self._slices[dim]

    def statistic(self, tokens: torch.Tensor) -> torch.Tensor:
        """Epps-Pulley statistic for token embeddings of shape (N, D)."""
        freqs = cast(torch.Tensor, self.freqs)
        slices = self._get_slices(tokens.size(1), tokens.device, tokens.dtype)
        z = tokens @ slices.t()  # (N, S)
        z = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-6)
        z = z.to(torch.float64)
        angles = z.unsqueeze(-1) * freqs.double()  # (N, S, F)
        phi_hat = torch.polar(torch.ones_like(angles), angles).mean(dim=0).abs() ** 2
        target = torch.exp(-(freqs.double() ** 2))
        return ((phi_hat - target) ** 2).sum(dim=1).mean()

    def forward(self, latents: dict) -> torch.Tensor:
        """Mean statistic over pyramid levels; latents maps level -> (B, D, T)."""
        b, d, t = latents.shape
        tokens = latents.transpose(1, 2).reshape(b * t, d)
        return self.statistic(tokens)

def find_similarity_loss(loss_name, device, use_cuda=False, temperature=1.0):
    if loss_name == 'dtw':
        return DTWLoss(device, use_cuda=use_cuda, gamma=temperature)
    elif loss_name == 'euclidean':
        return EuclideanDistanceLoss(temperature=temperature)
    else:
        raise ValueError(f"Unsupported similarity loss: {loss_name}")