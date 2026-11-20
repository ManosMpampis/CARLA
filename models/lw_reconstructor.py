"""Single-scale reconstruction LeWM models with time or frequency predictors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import Conv1dSamePadding, Conv2dSamePadding, ConvBlock1d, ConvBlock2d
from utils.masking_steered import TorchSubAnomalyMaskCollator
from utils.reconstruction_scores import normalize_score_mode, reconstruction_score_map


class MaskEngine(nn.Module):
    def __init__(self, masking):
        super().__init__()
        self.engine = TorchSubAnomalyMaskCollator(**masking)

    @staticmethod
    def _zero_mask_input(x, input_mask):
            m = input_mask.to(torch.bool)
            while m.ndim < x.ndim:
                m = m.unsqueeze(1)
            return x.masked_fill(m.expand_as(x), 0.0)
    
    def forward(self, x):
        return x, self.engine(x)

# Action
class TimeAuxiliary(nn.Module):
    """Convolution time auxiliary: localizes mask, feeds FiLM features.

    Action is never an input here (LeWM requirement); the mask head learns
    purely from Encoder(X_inj).
    """

    def __init__(self, in_dim: int, channels=(32, 32, 32),
                 kernels=(3, 5, 7), strides=1, act: str = "gelu", norm: str = "batch", 
                 dropout: float = 0.1):
        super().__init__()
        ks = list(kernels)
        chs = [in_dim] + list(channels)
        self.convs = nn.Sequential(
            *[ConvBlock1d(chs[i], chs[i + 1], kernel=ks[i], stride=strides,
                          act=act, norm=norm, dropout=dropout)
              for i in range(len(chs) - 1)]
        )
        self.mask_head = Conv1dSamePadding(chs[-1], 1, kernel_size=1)
        self.feat_dim = int(chs[-1])

    def forward(self, z):
        """Return (mask_logits (B, W), film_features (B, Da, W))."""
        h = self.convs(z)
        logits = self.mask_head(h).squeeze(1)
        return logits, h

# Action input to predictor mechanism
class IdentityFilm(nn.Module):
    """Identity FiLM: no time steering."""

    def __init__(self):
        super().__init__()

    def forward(self, h_f, aux_feat):
        return h_f


class TimeVaryingFiLM(nn.Module):
    """Per-timestep FiLM: aux time features -> gamma/beta broadcast over F."""

    def __init__(self, aux_dim: int, freq_dim: int):
        super().__init__()
        self.to_gamma = nn.Conv1d(aux_dim, freq_dim, kernel_size=1)
        self.to_beta = nn.Conv1d(aux_dim, freq_dim, kernel_size=1)
        for layer in (self.to_gamma, self.to_beta):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
        self.freq_dim = int(freq_dim)

    def forward(self, h_f, aux_feat):
        """Modulate (B, Df, T', F) by time-indexed (B, Da, W) features."""
        t_prime = h_f.size(2)
        a = F.interpolate(aux_feat, size=t_prime, mode="linear",
                          align_corners=False)
        gamma = self.to_gamma(a).unsqueeze(-1)
        beta = self.to_beta(a).unsqueeze(-1)
        return (1 + gamma) * h_f + beta


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
        a = F.interpolate(aux_feat, size=h.size(-1), mode="linear",
                          align_corners=False)
        return (1 + self.to_gamma(a)) * h + self.to_beta(a)

# Predictors
class TimePredictor(nn.Module):
    """Temporal convolution neck with the same three-scale FiLM interface."""

    def __init__(self, in_dim: int, stem_channels: int = 64,
                 neck_widths=(64, 64, 64), aux_dim: int = 32, act: str = "gelu", norm: str = "batch",
                 time_steering: bool = True, dropout: float = 0.1, n_fft: int = 0, hop_length: int = 0, win_length: int = 0):
        super().__init__()
        widths = [int(c) for c in neck_widths]
        if len(widths) != 3:
            raise ValueError("neck_widths must contain three channel counts")
        w0, w1, w2 = widths
        self.time_steering = bool(time_steering)
        self.stem_conv = ConvBlock1d(in_dim, stem_channels, kernel=7,
                                     dropout=dropout, act=act, norm=norm)
        self.in_proj = ConvBlock1d(stem_channels, w0, kernel=3, act=act, norm=norm, dropout=dropout)

        self.c0 = ConvBlock1d(w0, w1, kernel=3, stride=2, act=act, norm=norm, dropout=dropout)
        self.c1 = ConvBlock1d(w1, w2, kernel=3, stride=2, act=act, norm=norm, dropout=dropout)
        self.c2 = ConvBlock1d(w2, w2, kernel=3, act=act, norm=norm, dropout=dropout)
        self.up2 = ConvBlock1d(w2 + w1, w1, kernel=3, act=act, norm=norm, dropout=dropout)
        self.up1 = ConvBlock1d(w1 + w0, w0, kernel=3, act=act, norm=norm, dropout=dropout)
        self.down_p0 = ConvBlock1d(w0, w0, kernel=3, stride=2, act=act, norm=norm, dropout=dropout)
        self.pan1 = ConvBlock1d(w0 + w1, w1, kernel=3, act=act, norm=norm, dropout=dropout)
        self.final = ConvBlock1d(w1 + w0, w0, kernel=3, act=act, norm=norm, dropout=dropout)
        self.to_latent = Conv1dSamePadding(w0, in_dim, kernel_size=1)

        self.film0 = TimeFiLM(aux_dim, w1) if time_steering else IdentityFilm()
        self.film1 = TimeFiLM(aux_dim, w2) if time_steering else IdentityFilm()
        self.film2 = TimeFiLM(aux_dim, w2) if time_steering else IdentityFilm()

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
                 neck_widths=(64, 64, 64), aux_dim: int = 32, act: str = "gelu", norm: str = "batch",
                 dropout: float = 0.1, n_fft: int = 64, hop_length: int = 16, win_length: int = 64,
                 time_steering: bool = True):
        super().__init__()
        self.time_steering = time_steering
        self.in_dim = int(in_dim)
        self.stem_channels = int(stem_channels)
        self.stem_conv = ConvBlock1d(self.in_dim, self.stem_channels,
                                     kernel=7, stride=1, act="act", norm="batch",
                                     dropout=dropout)

        w = [int(c) for c in neck_widths]
        assert len(w) == 3
        self.neck_widths = w
        self.in_proj = ConvBlock2d(2 * self.stem_channels, w[0], act=act, norm=norm, dropout=dropout)
        self.c0 = ConvBlock2d(w[0], w[1], stride=2, act=act, norm=norm, dropout=dropout)
        self.c1 = ConvBlock2d(w[1], w[2], stride=2, act=act, norm=norm, dropout=dropout)
        self.c2 = ConvBlock2d(w[2], w[2], act=act, norm=norm, dropout=dropout)
        self.up2 = ConvBlock2d(w[2] + w[1], w[1], act=act, norm=norm, dropout=dropout)
        self.up1 = ConvBlock2d(w[1] + w[0], w[0], act=act, norm=norm, dropout=dropout)
        self.down_p0 = ConvBlock2d(w[0], w[0], stride=2, act=act, norm=norm, dropout=dropout)
        self.pan1 = ConvBlock2d(w[0] + w[1], w[1], act=act, norm=norm, dropout=dropout)
        self.final = ConvBlock2d(w[1] + w[0], w[0], act=act, norm=norm, dropout=dropout)
        self.out_proj = Conv2dSamePadding(w[0], 2 * self.stem_channels, kernel_size=1)
        self.to_latent = Conv1dSamePadding(self.stem_channels, self.in_dim, kernel_size=1)

        self.film0 = TimeVaryingFiLM(aux_dim, w[0]) if self.time_steering else IdentityFilm()
        self.film1 = TimeVaryingFiLM(aux_dim, w[1]) if self.time_steering else IdentityFilm()
        self.film2 = TimeVaryingFiLM(aux_dim, w[2]) if self.time_steering else IdentityFilm()

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
        window = torch.hann_window(win, device=h.device, dtype=z_tok.dtype)
        hf = h.reshape(b * self.stem_channels, w)

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


class LeWMModel(nn.Module):
    """Shared-encoder LeWM with selectable predictor and optional auxiliary."""

    def __init__(self, encoder: nn.Module, 
                 aux: dict = {"channels" : [32, 32, 32], "enable": True, "kernels": [7, 5, 3], "strides": 1, "norm": "batch", "act": "gelu", "dropout": 0.1},
                 predictor: dict = {"domain": "frequency", "time_steering": True, "stem_channels": 64, "neck_widths": [64, 64, 64], "n_fft": 64, "hop_length": 16, "win_length": 64},
                 engine: dict = {"min_ratio": 0.1, "max_ratio": 0.9, "portion_len": 0.99},
                 amp={"enabled": False, "dtype": torch.bfloat16},
                 score_mode="l1"):
        super().__init__()

        self.score_mode = normalize_score_mode(score_mode)
        self.amp = {"enabled":amp["enabled"], 
                    "dtype": torch.bfloat16 if (amp["dtype"] == "torch.bfloat16" or amp["dtype"] == torch.bfloat16) else torch.float16}

        self.engine = MaskEngine(engine) if engine is not None else None
        self.encoder = encoder
        input_dim = encoder.output_dims

        if predictor.get("domain", "") not in ("frequency", "time"):
            raise ValueError(f"Invalid predictor domain {predictor.get("domain", "")}")
        if predictor.get("time_steering", "") and not aux.get("enabled", ""):
            raise ValueError("predictor['time_steering'] requires aux['enabled']=True")

        aux_enabled = aux.pop("enabled", False)
        self.aux = TimeAuxiliary(input_dim, **aux) if aux_enabled else None

        predictor_cls = FreqPredictor if predictor.pop("domain") == "frequency" else TimePredictor

        predictor.update(
            in_dim=input_dim,
            aux_dim=self.aux.feat_dim if self.aux is not None else 0,
        )

        self.predictor = predictor_cls(**predictor)

        self.mask_token = nn.Parameter(torch.zeros(1, input_dim, 1))
        nn.init.normal_(self.mask_token, std=0.02)

        self.encoder_frozen = False

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self
    
    def _token_blend(self, z_inj, m_float):
        """Blend learned token into masked positions (hard if binary)."""
        blend = m_float.unsqueeze(1).expand_as(z_inj)
        tok = self.mask_token.expand_as(z_inj)
        return blend * tok + (1 - blend) * z_inj

    def encode(self, x):
        return self.encoder(x)

    def action(self, x):
        mask_logits, film_feat = self.aux(x)
        action = torch.sigmoid(mask_logits).to(x.dtype)
    
        #TODO: check if it is needed
        if action.size(-1) != x.size(-1):
            action = F.interpolate(action.unsqueeze(1),
                                    size=x.size(-1),
                                    mode="nearest").squeeze(1).to(x.dtype)
        return {"act": action, "film": film_feat, "act_output": mask_logits}

    def prediction(self, latent, action):
        z_tok = self._token_blend(latent, action["act"])
        return self.predictor(z_tok, action["film"])

    def fit(self, x):
        x, masked = self.engine(x)
        m = masked.get("input")
        x_inj = masked.get("X_inj")

        z_inj = self.encode(x_inj)
        z = self.encode(x)

        # Action
        act = self.act(z_inj)

        # Prediction
        pred = self.prediction(z_inj, act)
        return {
            "known_latents": z_inj,
            "unknown_latents": z,
            "predicted": pred,
            "action": act,
            "targets": {"reconstruction": z, "aux": m}
        }
    
    def forward(self, x):
        z = self.encode(x)
        act = self.action(z)
        pred = self.prediction(z, act)
        return {"known_latents": z, "prediction": pred}

    @torch.no_grad()
    def score(self, x):
        """Self-proposed-action anomaly evidence."""
        was_training = self.training
        self.eval()
        try:
            with torch.autocast(device_type=x.device.type, enabled=self.amp["enabled"], dtype=self.amp["dtype"]):
                out = self(x)
                score, _ = reconstruction_score_map(out["prediction"], out["known_latents"], mode=self.score_mode)
                return score
        finally:
            self.train(was_training)

    def latent_variance(self, latents):
        """Collapse diagnostic: mean per-dim variance."""
        z = latents
        return z.transpose(1, 2).reshape(-1, z.size(1)).var(dim=0).mean().item()