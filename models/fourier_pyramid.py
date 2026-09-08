"""Fourier-pyramid encoder (T4) with time→frequency FiLM steering (T5).

Time stream: the proven PyramidEncoder conv ladder over (B, C, W).
Frequency stream: per-level full-resolution FFT processing -- project the
raw input to F channels, pool to the level length, rFFT -> learned complex
mixing -> irFFT (FITS/FEDformer-style static frequency weights, T4).

Dynamic influence (T5): a time summary (global-pooled stem features, pure
time modality) generates per-level FiLM (gamma, beta) through
zero-initialized heads, so training starts from the unconditioned model
and the time pathway progressively steers the frequency layers. With
use_film=False the streams fuse with static weights only.

Geometry contract matches PyramidEncoder exactly (level_names/_dims/
_strides), so collators, criteria, and the Scorer work unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import StackedConvBlock
from models.convolutions import _init_weights
from models.encoder import _broadcast


class ComplexMix(nn.Module):
    """Learned static mixing across Fourier modes (real/imag as channels)."""

    def __init__(self, n_freq: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or n_freq * 2
        self.net = nn.Sequential(
            nn.Linear(2 * n_freq, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * n_freq))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                _init_weights(m)

    def forward(self, xf: torch.Tensor) -> torch.Tensor:
        """Mix (B, F, Freq) complex modes; returns same-shaped complex."""
        b, f, q = xf.shape
        cat = torch.cat([xf.real, xf.imag], dim=-1)
        out = self.net(cat)
        re, im = out.chunk(2, dim=-1)
        return torch.complex(re, im)


class FourierPyramidEncoder(nn.Module):
    """Dual-modality pyramid: conv time ladder + FFT frequency ladder."""

    def __init__(self, in_channels: int = 38, stem_channels: int = 32,
                 level_channels=(32, 64, 96), kernel_size=5, strides=(2, 2, 2),
                 depths=2, norm: str = "batch", dropout: bool = True,
                 stem_kernel: int = 7, freq_channels: int = 16,
                 use_freq: bool = True, use_film: bool = True,
                 film_hidden: int = 64, window: int = 256):
        super().__init__()
        n = len(level_channels)
        strides = list(strides)
        assert len(strides) == n
        kernels = _broadcast("kernel_size", kernel_size, n)
        depths = _broadcast("depths", depths, n)
        norms = _broadcast("norm", norm, n)

        # Time stream: proven stem + ladder.
        self.stem = StackedConvBlock(in_channels, stem_channels, depth=1,
                                     kernels=[int(stem_kernel)],
                                     stride=1, norm=norms[0], dropout=dropout)
        self.blocks = nn.ModuleList()
        ch_in = stem_channels
        dims = [stem_channels]
        for ch_out, k, s, d, nm in zip(level_channels, kernels, strides, depths, norms):
            kk = list(k) if isinstance(k, (list, tuple)) else [k] * int(d)
            self.blocks.append(StackedConvBlock(ch_in, int(ch_out), depth=int(d),
                                                kernels=kk, stride=int(s),
                                                norm=nm, dropout=dropout))
            ch_in = int(ch_out)
            dims.append(int(ch_out))

        self.level_names = ["L0"] + [f"L{i+1}" for i in range(n)]
        self.level_dims = dims
        acc, strides_out = 1, [1]
        for s in strides:
            acc *= int(s)
            strides_out.append(acc)
        self.level_strides = strides_out

        # Frequency stream (T4): raw-input projection + per-level FFT blocks.
        self.use_freq = bool(use_freq)
        self.use_film = bool(use_film) and bool(use_freq)
        self.freq_channels = int(freq_channels)
        if self.use_freq:
            self.freq_in = nn.Conv1d(in_channels, self.freq_channels, 1)
            _init_weights(self.freq_in)
            # Eager per-level mixers: lengths follow from the window and the
            # cumulative strides, so every parameter exists before the
            # optimizer is built (no lazy creation inside forward).
            self.freq_mixers = nn.ModuleList([
                ComplexMix((int(window) // s) // 2 + 1)
                for s in strides_out])
            self.fuse = nn.ModuleList([
                nn.Conv1d(d + self.freq_channels, d, 1)
                for d in dims])
            for m in self.fuse:
                _init_weights(m)

        # T5 steering: time summary -> per-level FiLM over freq activations.
        if self.use_film:
            self.film_trunk = nn.Sequential(
                nn.Linear(stem_channels, film_hidden), nn.GELU(),
                nn.Linear(film_hidden, film_hidden), nn.GELU())
            self.film_heads = nn.ModuleList([
                nn.Linear(film_hidden, 2 * self.freq_channels)
                for _ in dims])
            for h in self.film_heads:
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)

    def forward(self, x):
        """Encode a window into one fused feature map per level."""
        t0 = self.stem(x)
        t_feats = {"L0": t0}
        prev = t0
        for name, block in zip(self.level_names[1:], self.blocks):
            prev = block(prev)
            t_feats[name] = prev

        if not self.use_freq:
            return t_feats

        # T5: pure-time summary dislocated from every level's own input.
        summary = t0.mean(dim=-1)
        films = None
        if self.use_film:
            h = self.film_trunk(summary)
            films = [head(h) for head in self.film_heads]

        f = self.freq_in(x)
        feats = {}
        for i, name in enumerate(self.level_names):
            stride = self.level_strides[i]
            fl = f if stride == 1 else F.avg_pool1d(f, kernel_size=stride,
                                                   stride=stride)
            fl = fl[..., :t_feats[name].size(-1)]
            assert fl.size(-1) == int(x.size(-1)) // stride, \
                "FourierPyramidEncoder window differs from construction window"
            fb = torch.fft.irfft(self.freq_mixers[i](torch.fft.rfft(fl, dim=-1)),
                                 n=fl.size(-1), dim=-1)
            if films is not None:
                gamma, beta = films[i].chunk(2, dim=-1)
                fb = fb * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
            fused = torch.cat([t_feats[name], fb.to(t_feats[name].dtype)], dim=1)
            feats[name] = self.fuse[i](fused)
        return feats
