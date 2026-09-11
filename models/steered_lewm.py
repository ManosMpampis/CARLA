"""Time-steered frequency LeWM: single-scale ResNet encoder + time aux + freq predictor.

Terminology (spec `specs/lewm-time-steered-freq-predictor.md`):
- ResNet Encoder: time-domain only, two blocks of [7,5,3] + residual k=5.
- Convolution time auxiliary: 3-conv stack [7,5,3], sees only Encoder(X_inj).
  Never receives the action. Emits Time_mask logits + FiLM features.
- Predictor = frequency module only: mask-token-replaced latents -> stem ->
  STFT -> concat FPN/PAN neck -> iSTFT -> Z'. Sole action-conditioned path.
  Predictor input latents are mask-token-replaced (option 4): masked
  positions show a learned mask token instead of the corrupted values, so
  the mask tells the predictor what to reconstruct. The auxiliary still
  sees the raw corrupted latents (it needs the anomaly's signature to
  localize and steer).

Single scale: level_names ["L0"], level_strides [1], latents (B, D, W).
No stop-grad anywhere; anti-collapse is dual SIGReg in the criterion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ConvBlock1d
from models.convolutions import _init_weights

class IdentityFilm(nn.Module):
    """Identity FiLM: no time steering."""

    def __init__(self):
        super().__init__()

    def forward(self, h_f, aux_feat):
        return h_f

class SteeredResNetBlock(nn.Module):
    """One ResNet block: 3 main convs [7,5,3] + residual conv k=5."""

    def __init__(self, in_ch: int, out_ch: int, kernels=(7, 5, 3),
                 residual_kernel: int = 5, norm: str = "batch",
                 dropout: float = 0.0):
        super().__init__()
        ks = list(kernels)
        chs = [in_ch] + [out_ch] * len(ks)
        self.main = nn.Sequential(
            *[ConvBlock1d(chs[i], chs[i + 1], kernel=k, stride=1,
                          norm=norm, dropout=dropout)
              for i, k in enumerate(ks)]
        )
        self.residual = ConvBlock1d(in_ch, out_ch, kernel=int(residual_kernel),
                                    stride=1, norm=norm, dropout=dropout)
        self.act = nn.GELU()

    def forward(self, x):
        """Add residual branch to main branch, then activate."""
        return self.act(self.main(x) + self.residual(x))


class SteeredResNetEncoder(nn.Module):
    """Two-block time-domain encoder, length-preserving, single scale."""

    def __init__(self, in_channels: int = 38, enc_channels=(32, 64),
                 kernels=(7, 5, 3), residual_kernel: int = 5,
                 norm: str = "batch", dropout: bool = True):
        super().__init__()
        enc_channels = [in_channels] + list(enc_channels)
        self.blocks = nn.Sequential(
            *[SteeredResNetBlock(
                int(enc_channels[i]), int(enc_channels[i + 1]),
                kernels=kernels, residual_kernel=residual_kernel,
                norm=norm, dropout=dropout)
              for i in range(len(enc_channels) - 1)]
        )
        self.output_dims = enc_channels[-1]

    def forward(self, x):
        """Encode (B, C, W) -> (B, D, W)."""
        return self.blocks(x)


class TimeAuxiliary(nn.Module):
    """Convolution time auxiliary: localizes mask, feeds FiLM features.

    Action is never an input here (LeWM requirement); the mask head learns
    purely from Encoder(X_inj).
    """

    def __init__(self, in_dim: int, aux_channels=(32, 32, 32),
                 kernels=(7, 5, 3), norm: str = "batch",
                 dropout: bool = True):
        super().__init__()
        ks = list(kernels)
        chs = [in_dim] + list(aux_channels)
        self.convs = nn.Sequential(
            *[ConvBlock1d(chs[i], chs[i + 1], kernel=ks[i], stride=1,
                          norm=norm, dropout=dropout)
              for i in range(len(chs) - 1)]
        )
        self.mask_head = nn.Conv1d(chs[-1], 1, kernel_size=1)
        _init_weights(self.mask_head)
        self.feat_dim = int(chs[-1])

    def forward(self, z):
        """Return (mask_logits (B, W), film_features (B, Da, W))."""
        h = self.convs(z)
        logits = self.mask_head(h).squeeze(1)
        return logits, h


class _Conv2dBlock(nn.Module):
    """Conv2d + BatchNorm2d + GELU block for the TF-grid neck."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=kernel // 2),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(p=dropout, inplace=True) if dropout > 0.0 else nn.Identity()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _init_weights(m)
            elif isinstance(m, nn.BatchNorm2d):
                _init_weights(m)

    def forward(self, x):
        """Apply conv block."""
        return self.net(x)


class TimeVaryingFiLM(nn.Module):
    """Per-timestep FiLM: aux time features -> gamma/beta broadcast over F."""

    def __init__(self, aux_dim: int, freq_dim: int):
        super().__init__()
        self.to_gamma = nn.Conv1d(aux_dim, freq_dim, kernel_size=1)
        self.to_beta = nn.Conv1d(aux_dim, freq_dim, kernel_size=1)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)
        self.freq_dim = int(freq_dim)

    def forward(self, h_f, aux_feat):
        """Modulate (B, Df, T', F) by time-indexed (B, Da, W) features."""
        t_prime = h_f.size(2)
        a = F.interpolate(aux_feat.float(), size=t_prime, mode="linear",
                          align_corners=False)
        gamma = self.to_gamma(a).unsqueeze(-1)
        beta = self.to_beta(a).unsqueeze(-1)
        return (1 + gamma) * h_f + beta


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
                 neck_widths=(64, 64, 64), aux_dim: int = 32,
                 n_fft: int = 64, hop_length: int = 16, time_steering: bool = True,
                 win_length: int = 64, dropout: float = 0.1):
        super().__init__()
        self.time_steering = time_steering
        self.in_dim = int(in_dim)
        self.stem_channels = int(stem_channels)
        self.stem_conv = ConvBlock1d(self.in_dim, self.stem_channels,
                                     kernel=7, stride=1, norm="batch",
                                     dropout=dropout)

        w = [int(c) for c in neck_widths]
        assert len(w) == 3
        self.neck_widths = w
        self.in_proj = _Conv2dBlock(2 * self.stem_channels, w[0])
        self.c0 = _Conv2dBlock(w[0], w[1], stride=2, dropout=0.0)
        self.c1 = _Conv2dBlock(w[1], w[2], stride=2, dropout=0.0)
        self.c2 = _Conv2dBlock(w[2], w[2], dropout=0.0)
        self.up2 = _Conv2dBlock(w[2] + w[1], w[1], dropout=0.0)
        self.up1 = _Conv2dBlock(w[1] + w[0], w[0], dropout=0.0)
        self.down_p0 = _Conv2dBlock(w[0], w[0], stride=2, dropout=0.0)
        self.pan1 = _Conv2dBlock(w[0] + w[1], w[1], dropout=0.0)
        self.final = _Conv2dBlock(w[1] + w[0], w[0], dropout=0.0)
        self.out_proj = nn.Conv2d(w[0], 2 * self.stem_channels, kernel_size=1)
        _init_weights(self.out_proj)
        self.to_latent = nn.Conv1d(self.stem_channels, self.in_dim,
                                   kernel_size=1)
        _init_weights(self.to_latent)
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


class SteeredFreqLeWMModel(nn.Module):
    """Shared-encoder LeWM with time auxiliary + frequency predictor."""

    def __init__(self, encoder: nn.Module, aux_channels=(32, 32, 32),
                 stem_channels: int = 64, neck_widths=(64, 64, 64),
                 n_fft: int = 64, hop_length: int = 16,
                 win_length: int = 64, aux_kernels=(7, 5, 3), time_steering: bool = True,
                 norm: str = "batch", dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        input_dim = encoder.output_dims

        self.aux = TimeAuxiliary(input_dim, aux_channels=aux_channels,
                                 kernels=tuple(aux_kernels), norm=norm,
                                 dropout=dropout)

        self.predictor = FreqPredictor(
            input_dim, stem_channels=int(stem_channels),
            neck_widths=tuple(neck_widths), aux_dim=self.aux.feat_dim,
            n_fft=int(n_fft), hop_length=int(hop_length), time_steering=time_steering,
            win_length=int(win_length), dropout=dropout)

        # Learned unknown-token (option 4): marks what-to-reconstruct
        # positions in the predictor input. The auxiliary still sees raw
        # corrupted latents.
        self.mask_token = nn.Parameter(torch.zeros(1, input_dim, 1))
        nn.init.normal_(self.mask_token, std=0.02)

        self.level_names = ["L0"]
        self.level_dims = [int(input_dim)]
        self.level_strides = [1]
        self.target_encoder = None
        self.codebook = None
        self.anti_collapse = "sigreg"
        self.encoder_frozen = False

    @staticmethod
    def _zero_mask_input(x, input_mask):
        m = input_mask.to(torch.bool)
        while m.ndim < x.ndim:
            m = m.unsqueeze(1)
        return x.masked_fill(m.expand_as(x), 0.0)

    def encode(self, x):
        """Encode a window into single-scale latents (B, D, W)."""
        return self.encoder(x)

    def _token_blend(self, z_inj, m_float):
        """Blend learned token into masked positions (hard if binary)."""
        blend = m_float.unsqueeze(1).expand_as(z_inj)
        tok = self.mask_token.expand_as(z_inj)
        return blend * tok + (1 - blend) * z_inj

    def forward(self, x, mask=None, action=None):
        """Two streams, one encoder; no stop-grad on either stream.

        Predictor input latents are mask-token-blended: hard replacement on
        the binary train mask, soft blend with the proposed mask at
        inference. The auxiliary always sees raw corrupted latents.

        Action source follows the mode: train mode teacher-forces the given
        mask (learning signal); eval mode lets the auxiliary propose the
        action (detached), so validation loss scores the same inference
        path that test-time scoring uses. The aux-localization target
        (mask_target) always stays the given mask when one exists.
        """
        # Check for injected input mask (X_inj) first, then input mask, then action.
        if mask is not None and isinstance(mask, dict) and "X_inj" in mask:
            x_inj = mask["X_inj"].to(x.device, dtype=x.dtype)
            m = mask.get("input")
            m = torch.zeros(x.size(0), x.size(-1), device=x.device) \
                if m is None else m.to(x.device)
        elif mask is not None and isinstance(mask, dict) and "input" in mask:
            m = mask["input"].to(x.device)
            x_inj = self._zero_mask_input(x, m)
        elif action is not None:
            m = action.to(x.device) if torch.is_tensor(action) \
                else torch.zeros(x.size(0), x.size(-1), device=x.device)
            x_inj = x
        else:
            m = torch.zeros(x.size(0), x.size(-1), device=x.device)
            x_inj = x
        m_bool = m.to(torch.bool) if m.dtype == torch.bool \
            else (m > 0.5)

        z = self.encode(x)
        z_inj = self.encode(x_inj)
        mask_logits, film_feat = self.aux(z_inj)

        m_float = m.to(z.dtype).float()
        if m_float.ndim == 1:
            m_float = m_float.unsqueeze(0)
        if m_float.size(-1) != z.size(-1):
            m_float = F.interpolate(m_float.unsqueeze(1).float(),
                                    size=z.size(-1),
                                    mode="nearest").squeeze(1).to(z.dtype)

        z_tok = self._token_blend(z_inj, m_float)
        if not self.training:
            # Inference path (validation + scoring): aux proposes the action.
            prop = torch.sigmoid(mask_logits).detach().to(z.dtype)
            if prop.size(-1) != z.size(-1):
                prop = F.interpolate(prop.unsqueeze(1).float(),
                                     size=z.size(-1),
                                     mode="nearest").squeeze(1).to(z.dtype)
            z_tok = self._token_blend(z_inj, prop)
        z_pred = self.predictor(z_tok, film_feat)
        m_target = m_float.detach()
        return {
            "context": {"L0": z_inj},
            "latents": {"L0": z},
            "targets": {"L0": z},
            "predicted": {"L0": z_pred},
            "mask_logits": mask_logits,
            "mask_target": m_target,
            "mask": {"input": m_bool},
            "action": m_target,
        }

    def update_ema(self):
        """No teacher: no-op satisfying the Trainer contract."""

    def update_running_stats(self, latents):
        """No running statistics on the bare trunk."""

    def latent_variance(self, latents):
        """Collapse diagnostic: mean per-dim variance."""
        z = latents["L0"]
        return z.transpose(1, 2).reshape(-1, z.size(1)).var(dim=0).mean().item()

    @torch.no_grad()
    def score(self, x):
        """Self-proposed-action anomaly evidence fused to (B, W)."""
        was_training = self.training
        self.eval()
        z = self.encode(x.float())
        logits, feat = self.aux(z)
        m_hat = torch.sigmoid(logits)
        zp = self.predictor(self._token_blend(z, m_hat), feat)
        fused = (zp - z).abs().mean(dim=1)
        if was_training:
            self.train()
        return {"fused": fused, "levels": {"L0": fused}, "signals": {}}
