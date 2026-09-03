"""Metric-learning loss (H4): COCA invariance + variance guard + SAD push.

Normals cluster around a fixed per-level center (initialized from train
embeddings, never learned); synthetic anomalies are pushed out with an
inverse-distance term (Deep SAD). Variance hinge prevents collapse.
"""
import torch
import torch.nn as nn


class MetricLoss(nn.Module):
    """H4 criterion over L2-normalized embeddings {level: (B, E, T)}."""

    def __init__(self, lambda_var: float = 1.0, var_floor: float = 1.0,
                 eta_push: float = 1.0, eps: float = 1e-3):
        super().__init__()
        self.lambda_var = float(lambda_var)
        self.var_floor = float(var_floor)
        self.eta_push = float(eta_push)
        self.eps = float(eps)

    def forward(self, embeds: dict, centers: dict, anomaly_mask=None,
                raw_embeds: dict | None = None) -> dict:
        """Pull clean tokens to center; push synthetic-anomaly tokens away.

        The variance hinge reads unnormalized projections (`raw_embeds`);
        on L2-normalized inputs the floor would sit above any reachable
        std and degrade into a constant offset.
        """
        inv, var, push = [], [], []
        for n, e in embeds.items():
            c = centers[n].to(e.device, e.dtype)
            dist = ((e - c[None, :, None]) ** 2).sum(dim=1)  # (B, T)
            sim = 1 - dist / 4  # cosine-ish proximity on the hypersphere
            if anomaly_mask is not None and n in anomaly_mask:
                m = anomaly_mask[n].to(dist.dtype)
                inv.append(((2 - sim) * (1 - m)).mean())
                # Inverse-distance push on known synthetic anomalies.
                push.append((m / (dist + self.eps)).mean())
            else:
                inv.append((2 - sim).mean())
            # Variance hinge per embedding dim (anti-collapse, COCA-style).
            r = raw_embeds[n] if raw_embeds is not None and n in raw_embeds else e
            std = r.transpose(1, 2).reshape(-1, r.size(1)).std(dim=0)
            var.append(torch.relu(self.var_floor - std).mean())
        out = {"invariance": torch.stack(inv).mean(),
               "variance": torch.stack(var).mean()}
        out["push"] = torch.stack(push).mean() if push else inv[0].detach() * 0
        out["loss"] = out["invariance"] + self.lambda_var * out["variance"] \
            + self.eta_push * out["push"]
        return out
