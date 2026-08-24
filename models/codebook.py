import numpy as np
import torch
import torch.nn as nn


class SoftCodebook(nn.Module):
    """Learned prototype dictionary with soft-attention routing.

    Each pyramid level owns K prototypes. Tokens attend softly over
    prototypes (softmax over negative scaled distances); the quantization
    loss pulls prototypes toward the latent distribution and stabilizes
    training (SC-JEPA-style). At scoring time two extra signals come out:
    distance-to-nearest-prototype and routing attention entropy.

    Prototypes start as random unit-scaled vectors and are replaced by a
    k-means warmup initialization on first-epoch latents (see
    ``init_from_latents``).
    """

    def __init__(self, dims, num_prototypes: int = 64, temperature: float = 0.1):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.temperature = float(temperature)
        self.initialized = False
        # Levels are addressed by their canonical names L0..Ln.
        self.prototypes = nn.ParameterDict({
            f"L{i}": nn.Parameter(torch.randn(self.num_prototypes, dim) * 0.05)
            for i, dim in enumerate(dims)
        })

    def _proto(self, name: str) -> torch.Tensor:
        return self.prototypes[name]

    def route(self, name: str, z: torch.Tensor):
        """z: (B, D, T) -> distances (B, T, K), attention (B, T, K)."""
        proto = self._proto(name)
        b, d, t = z.shape
        tokens = z.transpose(1, 2).reshape(b * t, d)          # (N, D)
        dist = torch.cdist(tokens.unsqueeze(0), proto.unsqueeze(0)).squeeze(0)
        dist = dist.reshape(b, t, -1)                         # (B, T, K)
        attn = torch.softmax(-dist / (self.temperature + 1e-9), dim=-1)
        return dist, attn

    def quantization_loss(self, latents: dict) -> torch.Tensor:
        """Mean distance from each token to its attended prototypes."""
        values = []
        for name, z in latents.items():
            dist, attn = self.route(name, z)
            soft_dist = (dist * attn).sum(dim=-1)
            values.append(soft_dist.mean())
        return torch.stack(values).mean()

    @torch.no_grad()
    def signals(self, name: str, z: torch.Tensor):
        """Scoring-time anomaly evidence for one level."""
        dist, attn = self.route(name, z)
        nearest = dist.min(dim=-1).values                     # (B, T)
        entropy = -(attn.clamp_min(1e-9).log() * attn).sum(dim=-1)
        return nearest, entropy

    @torch.no_grad()
    def init_from_latents(self, latents_samples: dict, logger=None) -> None:
        """k-means warmup: replace random prototypes with cluster centers
        computed on collected first-epoch latents (numpy MiniBatchKMeans,
        deterministic seed)."""
        from sklearn.cluster import MiniBatchKMeans

        for name, chunks in latents_samples.items():
            z = torch.cat(chunks, dim=0)                      # (B..., D, T)
            b, d, t = z.shape
            tokens = z.transpose(1, 2).reshape(-1, d).cpu().numpy()
            k = min(self.num_prototypes, max(2, len(tokens)))
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=4, n_init="auto",
                                     batch_size=min(1024, len(tokens)))
            kmeans.fit(tokens)
            centers = torch.from_numpy(kmeans.cluster_centers_).to(
                device=self.prototypes[name].device,
                dtype=self.prototypes[name].dtype,
            )
            if k < self.num_prototypes:  # tiny-data guard: tile remaining slots
                reps = -(-self.num_prototypes // k)
                centers = centers.repeat(reps, 1)[:self.num_prototypes]
            self.prototypes[name].data.copy_(centers)
        self.initialized = True
        if logger is not None:
            logger.log("Codebook k-means warmup initialized "
                       f"({self.num_prototypes} prototypes/level)")
