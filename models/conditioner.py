"""Conditioning modules: disposable projector + action embedding.

Projector is training-only (LeWM requirement): SIGReg sees projected
tokens, scoring never does. ActionEmbed turns a box triple into the
action vector consumed by CondPredictor (closed-loop arm only).
"""
import torch
import torch.nn as nn

from models.convolutions import _init_weights


class Projector(nn.Module):
    """Small MLP + BatchNorm: Linear -> BN -> GELU -> Linear.

    BatchNorm is mandatory here: a final LayerNorm would block the
    Gaussianity the isotropy regularizer optimizes for (LeWM finding).
    """

    def __init__(self, dim: int, hidden=None):
        super().__init__()
        hidden = hidden or dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_weights(m)

    def forward(self, tokens):
        """Project (N, D) tokens; called per level, discarded at scoring."""
        return self.net(tokens)


class ActionEmbed(nn.Module):
    """Embed a YOLO-style triple (objectness, center, length) to action."""

    def __init__(self, action_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, action_dim))
        self.action_dim = action_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_weights(m)

    def forward(self, boxes):
        """Map box triples to action vectors; proposals stop here (sg).

        Detaching is the open-loop guarantee: the action path never trains
        the detector. The detector learns only from its own box loss.
        """
        if boxes.dim() == 3:  # (B, G, 3) grid -> soft top-1 over cells
            boxes = self.boxes_from_grid(boxes)
        return self.net(boxes.float().detach())

    @staticmethod
    def boxes_from_grid(grid):
        """Collapse a (B, G, 3) interval grid to one detached (B, 3) triple.

        Soft top-1 by objectness keeps the dominant proposal instead of
        averaging the whole grid into a meaningless mean box.
        """
        with torch.no_grad():
            weight = torch.softmax(grid[..., 0], dim=1).unsqueeze(-1)
            pooled = (grid * weight).sum(dim=1)
            pooled[..., 0] = torch.sigmoid(grid[..., 0]).max(dim=1).values
        return pooled.detach()
