"""Single-scale LeWM models with time or frequency predictors.

Terminology (spec `specs/lewm-time-steered-freq-predictor.md`):
- ResNet Encoder: time-domain only, two blocks of [7,5,3] + residual k=5.
- Convolution time auxiliary: 3-conv stack [7,5,3], sees only Encoder(X_inj).
  Never receives the action. Emits Time_mask logits + FiLM features.
- Predictor = a time-convolution or STFT-based frequency neck. Both accept
  mask-token-replaced latents and optional time-varying FiLM features.
  Predictor input latents are mask-token-replaced (option 4): masked
  positions show a learned mask token instead of the corrupted values, so
  the mask tells the predictor what to reconstruct. When present, the
  auxiliary sees the raw corrupted latents to localize and steer.

Single scale: level_names ["L0"], level_strides [1], latents (B, D, W).
No stop-grad anywhere; anti-collapse is dual SIGReg in the criterion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import Conv1dBlock, Conv2dBlock

    
class IdentityFiLM(nn.Module):
    """Identity FiLM: no time steering."""

    def __init__(self):
        super().__init__()

    def forward(self, h_f, aux_feat):
        return h_f


class TimeFiLM(nn.Module):
    """Per-position FiLM for a 1D temporal feature map."""

    def __init__(self, aux_dim: int, channels: int):
        super().__init__()
        self.to_gamma = nn.Conv1d(aux_dim, channels, 1)
        self.to_beta = nn.Conv1d(aux_dim, channels, 1)
        for layer in (self.to_gamma, self.to_beta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, h, aux_feat):
        a = F.interpolate(aux_feat.float(), size=h.size(-1), mode="linear",
                          align_corners=False)
        return (1 + self.to_gamma(a)) * h + self.to_beta(a)


class TimeVaryingFiLM(TimeFiLM):
    """Per-timestep FiLM: aux time features -> gamma/beta broadcast over F."""

    def __init__(self, aux_dim: int, freq_dim: int):
        super().__init__(aux_dim, freq_dim)
        self.freq_dim = int(freq_dim)

    def forward(self, h_f, aux_feat):
        """Modulate (B, Df, T', F) by time-indexed (B, Da, W) features."""
        t_prime = h_f.size(2)
        a = F.interpolate(aux_feat.float(), size=t_prime, mode="linear",
                          align_corners=False)
        gamma = self.to_gamma(a).unsqueeze(-1)
        beta = self.to_beta(a).unsqueeze(-1)
        return (1 + gamma) * h_f + beta


class TimePredictor(nn.Module):
    """Temporal convolution neck with the same three-scale FiLM interface."""

    def __init__(self, in_dim: int, stem_channels: int = 64,
                 neck_widths=(64, 64, 64), act: str = "relu", norm: str = "batch", dropout: float = 0.1):
        super().__init__()
        w = [int(c) for c in neck_widths]
        if len(w) != 3:
            raise ValueError("neck_widths must contain three channel counts")
        self.neck_widths = w
        self.stem_conv = Conv1dBlock(in_dim, stem_channels, kernel=7, act=act, norm=norm,
                                     dropout=dropout)
        self.in_proj = Conv1dBlock(stem_channels, w[0], kernel=3, act=act, norm=norm,
                                     dropout=dropout)
        self.c0 = Conv1dBlock(w[0], w[1], kernel=3, stride=2, act=act, norm=norm,
                                     dropout=dropout)
        self.c1 = Conv1dBlock(w[1], w[2], kernel=3, stride=2, act=act, norm=norm,
                                     dropout=dropout)
        self.c2 = Conv1dBlock(w[2], w[2], kernel=3, act=act, norm=norm,
                                     dropout=dropout)
        self.up2 = Conv1dBlock(w[2] + w[1], w[1], kernel=3, act=act, norm=norm,
                                     dropout=dropout)
        self.up1 = Conv1dBlock(w[1] + w[0], w[0], kernel=3, act=act, norm=norm,
                                     dropout=dropout)
        self.down_p0 = Conv1dBlock(w[0], w[0], kernel=3, stride=2, act=act, norm=norm,
                                     dropout=dropout)
        self.pan1 = Conv1dBlock(w[0] + w[1], w[1], kernel=3, act=act, norm=norm,
                                     dropout=dropout)
        self.final = Conv1dBlock(w[1] + w[0], w[0], kernel=3, act=act, norm=norm,
                                     dropout=dropout)
        self.to_latent = Conv1dBlock(w[0], in_dim, kernel_size=1, act=act, norm=norm, dropout=dropout)
        self.film0 = IdentityFiLM()
        self.film1 = IdentityFiLM()
        self.film2 = IdentityFiLM()

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


class TimePredictorFiLM(nn.Module):
    """Temporal convolution neck with the same three-scale FiLM interface."""

    def __init__(self, in_dim: int, stem_channels: int = 64,
                 neck_widths=(64, 64, 64), act: str = "relu", norm: str = "batch", dropout: float = 0.1):
        super().__init__(in_dim, stem_channels, neck_widths, act, norm, dropout)
        self.film0 = TimeFiLM(in_dim, self.neck_widths[1])
        self.film1 = TimeFiLM(in_dim, self.neck_widths[2])
        self.film2 = TimeFiLM(in_dim, self.neck_widths[2])


class FreqPredictor(nn.Module):
    """Frequency-only predictor: token-masked latents -> STFT -> neck -> iSTFT.

    The what-to-reconstruct signal is mask-token replacement applied by the
    model before entry (option 4): masked latent positions already show the
    learned token, so the stem takes latents alone with no mask concat.
    STFT/iSTFT run in float32 for AMP safety; the neck is real-valued 2D
    convs on stacked real/imag channels with YOLO-style concat fusion:
    strided down path, top-down upsample+concat path, bottom-up PAN path.
    """

    def __init__(self, in_dim: int, stem_channels: int = 64,
                 neck_widths=(64, 64, 64),
                 n_fft: int = 64, hop_length: int = 16,
                 win_length: int = 64, act: str = "relu",
                 norm: str = "batch", dropout: float = 0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.stem_channels = int(stem_channels)
        self.stem_conv = Conv1dBlock(self.in_dim, self.stem_channels,
                                     kernel=7, stride=1, act=act, norm=norm,
                                     dropout=dropout)

        w = [int(c) for c in neck_widths]
        assert len(w) == 3
        self.neck_widths = w
        self.in_proj = Conv2dBlock(2 * self.stem_channels, w[0], act=act,
                                     dropout=dropout)
        self.c0 = Conv2dBlock(w[0], w[1], stride=2, act=act,
                                     dropout=dropout)
        self.c1 = Conv2dBlock(w[1], w[2], stride=2, act=act,
                                     dropout=dropout)
        self.c2 = Conv2dBlock(w[2], w[2], act=act,
                                     dropout=dropout)

        self.up2 = Conv2dBlock(w[2] + w[1], w[1], act=act,
                                     dropout=dropout)
        self.up1 = Conv2dBlock(w[1] + w[0], w[0], act=act,
                                     dropout=dropout)
        
        self.down_p0 = Conv2dBlock(w[0], w[0], stride=2, act=act,
                                     dropout=dropout)
        self.pan1 = Conv2dBlock(w[0] + w[1], w[1], act=act,
                                     dropout=dropout)
        self.final = Conv2dBlock(w[1] + w[0], w[0], act=act,
                                     dropout=dropout)
        
        self.out_proj = Conv2dBlock(w[0], 2 * self.stem_channels, kernel_size=1, act=act,
                                     dropout=dropout)
        self.to_latent = Conv1dBlock(self.stem_channels, self.in_dim,
                                   kernel_size=1, act=act, norm=norm,
                                   dropout=dropout)
        self.film0 = IdentityFiLM()
        self.film1 = IdentityFiLM()
        self.film2 = IdentityFiLM()

        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)

    def _resolve_stft(self, w: int):
        n_fft = max(8, min(self.n_fft, int(w)))
        win = max(4, min(self.win_length, n_fft))
        hop = max(1, min(self.hop_length, max(n_fft // 4, 1)))
        return n_fft, hop, win

    def forward(self, z_tok, aux_feat):
        """Map token-blended latents (B,D,W) + aux (B,Da,W) -> Z' (B,D,W)."""
        b, _, w = z_tok.shape
        h = self.stem_conv(z_tok)

        n_fft, hop, win = self._resolve_stft(w)
        window = torch.hann_window(win, device=h.device, dtype=torch.float32)
        hf = h.float().reshape(b * self.stem_channels, w)

        spec = torch.stft(hf, n_fft=n_fft, hop_length=hop, win_length=win,
                          window=window, center=True, return_complex=True)
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

        real, imag = out.chunk(2, dim=1)
        cout = torch.complex(real, imag).reshape(b * self.stem_channels,
                                                 t_prime, n_freq)
        cout = cout.permute(0, 2, 1).contiguous()
        rec = torch.istft(cout, n_fft=n_fft, hop_length=hop, win_length=win,
                          window=window, center=True, length=w)
        rec = rec.reshape(b, self.stem_channels, w).to(z_tok.dtype)
        return self.to_latent(rec)

    @torch.no_grad()
    def score(self, x):
        """Self-proposed-action anomaly evidence fused to (B, W)."""
        was_training = self.training
        self.eval()
        z = self.encoder(x.float())
        if self.aux is None:
            feat = None
            m_hat = z.new_zeros(z.size(0), z.size(-1))
        else:
            logits, feat = self.aux(z)
            m_hat = torch.sigmoid(logits)
        zp = self.predictor(self._token_blend(z, m_hat), feat)
        fused = (zp - z).abs().mean(dim=1)
        if was_training:
            self.train()
        return {"fused": fused, "levels": {"L0": fused}, "signals": {}}


class FreqPredictorFiLM(FreqPredictor):
    """Frequency-only predictor: token-masked latents -> STFT -> neck -> iSTFT.

    The what-to-reconstruct signal is mask-token replacement applied by the
    model before entry (option 4): masked latent positions already show the
    learned token, so the stem takes latents alone with no mask concat.
    STFT/iSTFT run in float32 for AMP safety; the neck is real-valued 2D
    convs on stacked real/imag channels with YOLO-style concat fusion:
    strided down path, top-down upsample+concat path, bottom-up PAN path.
    """

    def __init__(self, in_dim: int, stem_channels: int = 64,
                 neck_widths=(64, 64, 64),
                 n_fft: int = 64, hop_length: int = 16,
                 win_length: int = 64, act: str = "relu", norm: str = "batch", dropout: float = 0.1):
        super().__init__(in_dim, stem_channels, neck_widths, n_fft, hop_length,
                        win_length, act, norm, dropout)

        self.film0 = TimeVaryingFiLM(in_dim, self.neck_widths[0]) 
        self.film1 = TimeVaryingFiLM(in_dim, self.neck_widths[1])
        self.film2 = TimeVaryingFiLM(in_dim, self.neck_widths[2])