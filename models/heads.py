"""Pluggable heads: auxiliary focus heads + H1..H4 scoring heads.

All heads read trunk maps {level: (B, D, T)} and emit per-position maps.
Auxiliary heads are training-only (pruned at inference). H-heads emit
the dense channels the scorer fuses. Each head is independently runnable.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convolutions import _init_weights


def _head_conv(dim: int, out: int) -> nn.Conv1d:
    conv = nn.Conv1d(dim, out, 1)
    _init_weights(conv)
    return conv


class ReconAuxHead(nn.Module):
    """Lightweight training-only reconstructor: pooled trunk -> window.

    Forces the time pathway to keep fine + global detail. Small by design
    (bottleneck) so it regularizes instead of memorizing.
    """

    def __init__(self, dims, out_channels: int, bottleneck: int = 16):
        super().__init__()
        self.proj = nn.ModuleDict({n: _head_conv(d, bottleneck) for n, d in dims.items()})
        self.fuse = nn.Conv1d(bottleneck * len(dims), out_channels, 1)
        _init_weights(self.fuse)

    def forward(self, feats: dict, window: int):
        """Upsample each level to `window` steps, fuse to (B, C, W)."""
        ups = [F.interpolate(self.proj[n](feats[n]), size=window, mode="linear",
                             align_corners=False) for n in feats]
        return self.fuse(torch.cat(ups, dim=1))


class BoxAuxHead(nn.Module):
    """YOLO-1D auxiliary head: per-cell (objectness, center, length)."""

    def __init__(self, dim: int, num_cells: int = 16, num_types: int = 0):
        super().__init__()
        self.num_cells = int(num_cells)
        self.out_dim = 3 + int(num_types)
        self.conv = _head_conv(dim, 64)
        self.grid = nn.Conv1d(64, self.out_dim, 1)
        _init_weights(self.grid)

    def forward(self, feat):
        """Map (B, D, T) to (B, G, 3+types) interval grid."""
        h = torch.relu(self.conv(feat))
        # Adaptive pool to fixed grid so loss geometry is resolution-free.
        h = F.adaptive_avg_pool1d(h, self.num_cells).transpose(1, 2)
        return self.grid(h.transpose(1, 2)).transpose(1, 2)


class H1DetectHead(nn.Module):
    """Detection head: dense per-position logit + shared interval grid."""

    def __init__(self, dims, num_cells: int = 16):
        super().__init__()
        coarse = max(dims, key=lambda n: dims[n])
        self.logits = nn.ModuleDict({n: _head_conv(d, 1) for n, d in dims.items()})
        self.boxes = BoxAuxHead(dims[coarse], num_cells=num_cells)

    def forward(self, feats: dict):
        """Return {'logit': {level: (B, T)}, 'boxes': (B, G, 3)}."""
        return {"logit": {n: self.logits[n](feats[n]).squeeze(1) for n in feats},
                "boxes": self.boxes(feats[max(feats, key=lambda n: feats[n].shape[1])])}


class H2ReconHead(nn.Module):
    """Reconstruction head: per-level decoders + optional tiny VAE.

    Tiny VAE heads (mu/logvar per level) stay OFF by default; enable per
    arm with KL annealing + free-bits in the criterion.
    """

    def __init__(self, dims, out_channels: int, tiny_vae: bool = False,
                 vae_dim: int = 8):
        super().__init__()
        self.decoders = nn.ModuleDict({n: _head_conv(d, out_channels) for n, d in dims.items()})
        self.tiny_vae = bool(tiny_vae)
        self.mu = self.logvar = None
        if tiny_vae:
            self.mu = nn.ModuleDict({n: _head_conv(d, vae_dim) for n, d in dims.items()})
            self.logvar = nn.ModuleDict({n: _head_conv(d, vae_dim) for n, d in dims.items()})

    def forward(self, feats: dict, window: int):
        """Decode each level to (B, C, W); VAE params if enabled."""
        out = {"recon": {n: F.interpolate(self.decoders[n](feats[n]), size=window,
                                          mode="linear", align_corners=False) for n in feats}}
        if self.tiny_vae:
            assert self.mu is not None and self.logvar is not None
            out["mu"] = {n: self.mu[n](feats[n]) for n in feats}
            out["logvar"] = {n: self.logvar[n](feats[n]) for n in feats}
        return out


class H3EnergyHead(nn.Module):
    """Scalar energy per position: low for normal, margin above otherwise."""

    def __init__(self, dims, hidden: int = 32):
        super().__init__()
        self.nets = nn.ModuleDict({
            n: nn.Sequential(_head_conv(d, hidden), nn.GELU(), _head_conv(hidden, 1))
            for n, d in dims.items()})

    def forward(self, feats: dict):
        """Return {level: (B, T)} energy maps."""
        return {n: self.nets[n](feats[n]).squeeze(1) for n in feats}


class H4MetricHead(nn.Module):
    """Embedding head for distance-to-center scoring (per level)."""

    def __init__(self, dims, embed_dim: int = 32):
        super().__init__()
        self.proj = nn.ModuleDict({n: _head_conv(d, embed_dim) for n, d in dims.items()})
        # Fixed per-level centers (buffers, set from train embeddings, never learned).
        for n, d in dims.items():
            self.register_buffer(f"center_{n}", torch.zeros(embed_dim))
        self.embed_dim = embed_dim
        self._dims = dict(dims)

    def centers(self):
        """Read current fixed centers as {level: (D,)}."""
        return {n: getattr(self, f"center_{n}") for n in self._dims}

    @torch.no_grad()
    def init_centers(self, feats: dict):
        """Set each center to the mean embedding (call once on train data)."""
        with torch.no_grad():
            for n, z in feats.items():
                e = self.proj[n](z).mean(dim=(0, 2))
                e = e / (e.norm() + 1e-9)
                getattr(self, f"center_{n}").copy_(e)

    def forward(self, feats: dict):
        """Return L2-normalized embeddings {level: (B, E, T)}."""
        return {n: F.normalize(self.raw(feats)[n], dim=1, eps=1e-9)
                for n in feats}

    def raw(self, feats: dict):
        """Unnormalized projections (for the variance anti-collapse term)."""
        return {n: self.proj[n](z) for n, z in feats.items()}

    def distances(self, feats: dict):
        """Squared distance to center per position: {level: (B, T)}."""
        return {n: ((e - self.centers()[n][None, :, None]) ** 2).sum(dim=1)
                for n, e in self.forward(feats).items()}
