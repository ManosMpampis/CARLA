"""Time-domain and STFT predictors for length-preserving latent reconstruction."""

import torch
from torch import nn
from torch.nn import functional as F

from models.blocks import Conv1dBlock, Conv2dBlock, _init_weights


class IdentityFiLM(nn.Module):
    def forward(self, features, aux_feat):
        return features


class ReconstructionPredictor(nn.Module):
    def predict(self, current, action, *, target_tokens, query_start=0):
        if target_tokens != current.size(-1):
            raise ValueError("reconstruction predictor requires equal current and target token counts")
        return self(action.token_input, action.film_feat)


class TimeFiLM(nn.Module):
    """Apply temporal FiLM to a 1D feature map or broadcast it over frequency."""

    def __init__(self, aux_dim: int, channels: int):
        super().__init__()
        self.to_gamma = nn.Conv1d(aux_dim, channels, 1)
        self.to_beta = nn.Conv1d(aux_dim, channels, 1)
        for layer in (self.to_gamma, self.to_beta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features, aux_feat):
        aux = F.interpolate(aux_feat.float(), size=features.size(2), mode="linear",
                            align_corners=False)
        gamma, beta = self.to_gamma(aux), self.to_beta(aux)
        if features.ndim == 4:
            gamma, beta = gamma.unsqueeze(-1), beta.unsqueeze(-1)
        return (1 + gamma) * features + beta


class TimePredictor(ReconstructionPredictor):
    """Temporal convolution neck with the same three-scale FiLM interface."""

    def __init__(self, in_dim: int, stem_channels: int = 64,
                 neck_widths=(64, 64, 64), aux_dim: int = 32,
                 time_steering: bool = True, dropout: float = 0.1):
        super().__init__()
        widths = [int(c) for c in neck_widths]
        if len(widths) != 3:
            raise ValueError("neck_widths must contain three channel counts")
        w0, w1, w2 = widths
        self.time_steering = bool(time_steering)
        self.stem_conv = Conv1dBlock(in_dim, stem_channels, kernel=7,
                                     dropout=dropout)
        self.in_proj = Conv1dBlock(stem_channels, w0, kernel=3)
        self.c0 = Conv1dBlock(w0, w1, kernel=3, stride=2)
        self.c1 = Conv1dBlock(w1, w2, kernel=3, stride=2)
        self.c2 = Conv1dBlock(w2, w2, kernel=3)
        self.up2 = Conv1dBlock(w2 + w1, w1, kernel=3)
        self.up1 = Conv1dBlock(w1 + w0, w0, kernel=3)
        self.down_p0 = Conv1dBlock(w0, w0, kernel=3, stride=2)
        self.pan1 = Conv1dBlock(w0 + w1, w1, kernel=3)
        self.final = Conv1dBlock(w1 + w0, w0, kernel=3)
        self.to_latent = nn.Conv1d(w0, in_dim, kernel_size=1)
        _init_weights(self.to_latent)
        self.film0 = TimeFiLM(aux_dim, w1) if time_steering else IdentityFiLM()
        self.film1 = TimeFiLM(aux_dim, w2) if time_steering else IdentityFiLM()
        self.film2 = TimeFiLM(aux_dim, w2) if time_steering else IdentityFiLM()

    def forward(self, z_tok, aux_feat):
        g = self.in_proj(self.stem_conv(z_tok))
        d1 = self.film0(self.c0(g), aux_feat)
        d2 = self.film1(self.c1(d1), aux_feat)
        p2 = self.film2(self.c2(d2), aux_feat)
        p1 = self.up2(torch.cat([
            F.interpolate(p2, size=d1.size(-1), mode="nearest"), d1], dim=1))
        p0 = self.up1(torch.cat([
            F.interpolate(p1, size=g.size(-1), mode="nearest"), g], dim=1))
        n1 = self.pan1(torch.cat([self.down_p0(p0), p1], dim=1))
        n0 = self.final(torch.cat([
            F.interpolate(n1, size=p0.size(-1), mode="nearest"), p0], dim=1))
        return self.to_latent(n0)


class FreqPredictor(ReconstructionPredictor):
    """Frequency-only predictor: token-masked latents -> STFT -> neck -> iSTFT.

    The what-to-reconstruct signal is mask-token replacement applied by the
    model before entry (option 4): masked latent positions already show the
    learned token, so the stem takes latents alone with no mask concat.
    STFT/iSTFT run in float32 for AMP safety; the neck is real-valued 2D
    convs on stacked real/imag channels with YOLO-style concat fusion:
    strided down path, top-down upsample+concat path, bottom-up PAN path.
    """

    def __init__(self, in_dim: int, stem_channels: int = 64,
                 neck_widths=(64, 64, 64), aux_dim: int = 32,
                 n_fft: int = 64, hop_length: int = 16, time_steering: bool = True,
                 win_length: int = 64, dropout: float = 0.1):
        super().__init__()
        self.time_steering = time_steering
        self.in_dim = int(in_dim)
        self.stem_channels = int(stem_channels)
        self.stem_conv = Conv1dBlock(self.in_dim, self.stem_channels,
                                     kernel=7, stride=1, norm="batch",
                                     dropout=dropout)

        w = [int(c) for c in neck_widths]
        if len(w) != 3:
            raise ValueError("neck_widths must contain three channel counts")
        self.neck_widths = w
        self.in_proj = Conv2dBlock(2 * self.stem_channels, w[0])
        self.c0 = Conv2dBlock(w[0], w[1], stride=2, dropout=0.0)
        self.c1 = Conv2dBlock(w[1], w[2], stride=2, dropout=0.0)
        self.c2 = Conv2dBlock(w[2], w[2], dropout=0.0)
        self.up2 = Conv2dBlock(w[2] + w[1], w[1], dropout=0.0)
        self.up1 = Conv2dBlock(w[1] + w[0], w[0], dropout=0.0)
        self.down_p0 = Conv2dBlock(w[0], w[0], stride=2, dropout=0.0)
        self.pan1 = Conv2dBlock(w[0] + w[1], w[1], dropout=0.0)
        self.final = Conv2dBlock(w[1] + w[0], w[0], dropout=0.0)
        self.out_proj = nn.Conv2d(w[0], 2 * self.stem_channels, kernel_size=1)
        _init_weights(self.out_proj)
        self.to_latent = nn.Conv1d(self.stem_channels, self.in_dim,
                                   kernel_size=1)
        _init_weights(self.to_latent)
        self.film0 = TimeFiLM(aux_dim, w[1]) if self.time_steering else IdentityFiLM()
        self.film1 = TimeFiLM(aux_dim, w[2]) if self.time_steering else IdentityFiLM()
        self.film2 = TimeFiLM(aux_dim, w[2]) if self.time_steering else IdentityFiLM()

        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)

    def _resolve_stft(self, w: int):
        n_fft = max(8, min(self.n_fft, int(w)))
        win = max(4, min(self.win_length, n_fft))
        hop = max(1, min(self.hop_length, max(win // 4, 1)))
        return n_fft, hop, win

    def forward(self, z_tok, aux_feat):
        """Map token-blended latents (B,D,W) + aux (B,Da,W) -> Z' (B,D,W)."""
        b, _, w = z_tok.shape
        h = self.stem_conv(z_tok)

        n_fft, hop, win = self._resolve_stft(w)
        window = torch.hann_window(win, device=h.device, dtype=torch.float32)
        hf = h.float().reshape(b * self.stem_channels, w)

        with torch.autocast(device_type=h.device.type, enabled=False):
            spec = torch.stft(hf, n_fft=n_fft, hop_length=hop, win_length=win,
                              window=window, center=True,
                              pad_mode="reflect" if w > n_fft // 2 else "constant",
                              return_complex=True)
        # spec: (B*S, F, T') -> (B, 2S, T', F)
        spec = spec.permute(0, 2, 1).contiguous()
        t_prime, n_freq = spec.size(1), spec.size(2)
        tf = torch.stack([spec.real, spec.imag], dim=2).reshape(
            b, 2 * self.stem_channels, t_prime, n_freq)

        # Down path (full -> /2 -> /4 -> /4), FiLM-steered per scale.
        g = self.in_proj(tf)
        d1 = self.film0(self.c0(g), aux_feat)
        d2 = self.film1(self.c1(d1), aux_feat)
        p2 = self.film2(self.c2(d2), aux_feat)
        # Top-down path with concats.
        p1 = self.up2(torch.cat(
            [F.interpolate(p2, size=d1.shape[2:], mode="nearest"), d1], dim=1))
        p0 = self.up1(torch.cat(
            [F.interpolate(p1, size=g.shape[2:], mode="nearest"), g], dim=1))
        # Bottom-up PAN path with concats, fused back to full resolution.
        n1 = self.pan1(torch.cat([self.down_p0(p0), p1], dim=1))
        n0 = self.final(torch.cat(
            [F.interpolate(n1, size=p0.shape[2:], mode="nearest"), p0], dim=1))
        out = self.out_proj(n0)

        real, imag = out.float().chunk(2, dim=1)
        cout = torch.complex(real, imag).reshape(b * self.stem_channels,
                                                 t_prime, n_freq)
        cout = cout.permute(0, 2, 1).contiguous()
        with torch.autocast(device_type=h.device.type, enabled=False):
            rec = torch.istft(cout, n_fft=n_fft, hop_length=hop, win_length=win,
                              window=window, center=True, length=w)
        rec = rec.reshape(b, self.stem_channels, w).to(z_tok.dtype)
        return self.to_latent(rec)
