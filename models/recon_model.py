"""Phase-2 reconstruction model: frozen-able steered encoder + mirrored head.

Default inference (grilled Q6) is encoder + head only: fully convolutional,
timestep-agnostic for any W with W % S == 0, score = mean_C |x - x_hat|.

Aux-crop inference (grilled Q10) reuses the frozen pretext TimeAuxiliary:
Z = encoder(X) is cropped at latent resolution where sigmoid(mask_logits)
> 0.5 (connected components), each Z[:,:,l:r] goes straight through the
same head, crop errors are filed back at input steps [l*S, r*S), everything
else scores 0. Empty proposal -> all zeros (classified normal, Q10c).
Training is always on full normal windows (Q10b).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _connected_components_1d(flags):
    """1D True-runs -> [(l, r)] with r exclusive (python ints)."""
    runs, start = [], None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((int(start), int(i)))
            start = None
    if start is not None:
        runs.append((int(start), int(len(flags))))
    return runs


class ReconModel(nn.Module):
    """Encoder + mirrored recon head (+ optional frozen aux for aux-crop)."""

    def __init__(self, encoder: nn.Module, head: nn.Module, aux=None):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.aux = aux
        self.level_names = ["L0"]
        lat_d = int(getattr(encoder, "output_dims", 0))
        self.level_dims = [lat_d]
        self.level_strides = [int(getattr(encoder, "total_stride", 1))]
        self.target_encoder = None
        self.codebook = None
        self.anti_collapse = "none"
        self.encoder_frozen = False

    @property
    def total_stride(self):
        """Total downsampling S (Phase-2 contract divisor)."""
        return int(self.level_strides[0])

    def encode(self, x):
        """Encode (B, C, W) -> (B, D, W/S)."""
        return self.encoder(x)

    def reconstruct(self, x):
        """Full-window recon (B, C, W)."""
        return self.head(self.encode(x))

    def forward(self, x, mask=None, action=None):
        """Trainer contract: dense recon of the (clean, normal-only) input."""
        z = self.encode(x)
        x_hat = self.head(z)
        return {
            "recon": x_hat,
            "target": x,
            "context": {"L0": z},
            "latents": {"L0": z},
            "targets": {"L0": z},
            "predicted": {"L0": z},
            "mask": None,
            "action": None,
        }

    def update_ema(self):
        """No teacher: no-op satisfying the Trainer contract."""

    def update_running_stats(self, latents):
        """No running statistics on the recon model."""

    def latent_variance(self, latents):
        """Collapse diagnostic: mean per-dim variance."""
        z = latents["L0"]
        return z.transpose(1, 2).reshape(-1, z.size(1)).var(dim=0).mean().item()

    # -- scoring (fp32, eval mode) --------------------------------------
    @torch.no_grad()
    def score(self, x, aux_crop=None):
        """Anomaly evidence fused to (B, W).

        Full mode: mean_C |x - head(encoder(x))|.
        Aux-crop mode: latent crops from aux proposals, zeros elsewhere,
        zeros everywhere when the aux proposes nothing (Q10c).
        ``aux_crop=None`` (the Scorer path) reads ``self.score_aux_crop``.
        """
        if aux_crop is None:
            aux_crop = bool(getattr(self, "score_aux_crop", False))
        was_training = self.training
        self.eval()
        x = x.float()
        if not aux_crop:
            x_hat = self.head(self.encode(x))
            errors = (x_hat - x).abs()
            fused = errors.mean(dim=1)
            out = {"fused": fused, "levels": {"L0": fused},
                   "signals": {f"channel/{i}": errors[:, i]
                                for i in range(errors.shape[1])}}
            if was_training:
                self.train()
            return out
        if self.aux is None:
            raise ValueError("aux_crop=True needs aux (with_aux=true + "
                             "pretrained aux weights)")
        s = self.total_stride
        z = self.encode(x)
        logits, _ = self.aux(z)
        prob = torch.sigmoid(logits).detach()
        b, w = x.shape[0], x.shape[-1]
        fused = x.new_zeros((b, w))
        dimensions = x.new_zeros((b, x.shape[1], w))
        for i in range(b):
            flags = (prob[i] > 0.5).detach().cpu().tolist()
            for (l, r) in _connected_components_1d(flags):
                if r <= l:
                    continue
                z_crop = z[i:i + 1, :, l:r]
                x_hat_crop = self.head(z_crop.float())
                # Target input segment mirrored by the stride mapping.
                tgt = x[i:i + 1, :, l * s:r * s]
                if x_hat_crop.shape[-1] != tgt.shape[-1]:
                    # Defensive: shapes match exactly when W % S == 0 and the
                    # head mirrors the encoder rates; never crop silently in
                    # the default contract, just surface the mismatch.
                    raise RuntimeError(
                        f"recon crop len {x_hat_crop.shape[-1]} != target "
                        f"len {tgt.shape[-1]} (l={l}, r={r}, S={s})")
                errors = (x_hat_crop - tgt).abs().squeeze(0)
                dimensions[i, :, l * s:r * s] = errors.to(dimensions.dtype)
                fused[i, l * s:r * s] = errors.mean(dim=0).to(fused.dtype)
        if was_training:
            self.train()
        fused = fused.to(torch.float32)
        return {"fused": fused, "levels": {"L0": fused},
                "signals": {f"channel/{i}": dimensions[:, i]
                             for i in range(dimensions.shape[1])}}

    def score_with_flag(self, x):
        """Scorer contract wrapper (reads self.score_aux_crop at call time)."""
        return self.score(x, aux_crop=bool(getattr(self, "score_aux_crop", False)))
