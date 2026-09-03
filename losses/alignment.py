"""Energy/margin loss (H3) and two-view KL alignment loss (O2 channel)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyLoss(nn.Module):
    """Margin-ranked scalar energy: clean low, synthetic above margin."""

    def __init__(self, margin: float = 5.0):
        super().__init__()
        self.margin = float(margin)

    def forward(self, energy: dict, anomaly_mask=None) -> dict:
        """Penalize clean energy magnitude + margin violations on probes."""
        terms = []
        for n, e in energy.items():
            if anomaly_mask is not None and n in anomaly_mask:
                m = anomaly_mask[n].to(e.dtype)
                clean = (e.abs() * (1 - m)).mean()
                viol = (F.relu(self.margin - e) * m).mean()
                terms.append(clean + viol)
            else:
                terms.append(e.abs().mean())
        loss = torch.stack(terms).mean()
        return {"loss": loss, "energy": loss.detach()}


class ViewKLLoss(nn.Module):
    """Symmetric KL between two views with stop-gradient (DCdetector-style).

    Same timestamp in two contexts should share an embedding; anomalies
    disagree. No decoder needed; the KL map itself is a score channel.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _log_prob(x):
        return F.log_softmax(x, dim=1)

    def forward(self, view_p: dict, view_n: dict) -> dict:
        """Mean 0.5*KL(P||sg(N)) + 0.5*KL(N||sg(P)) over levels/positions."""
        terms = []
        for n in view_p:
            p, q = view_p[n], view_n[n]
            kl1 = F.kl_div(self._log_prob(p), q.detach().softmax(dim=1),
                           reduction="none").sum(dim=1).mean()
            kl2 = F.kl_div(self._log_prob(q), p.detach().softmax(dim=1),
                           reduction="none").sum(dim=1).mean()
            terms.append(0.5 * (kl1 + kl2))
        loss = torch.stack(terms).mean()
        return {"loss": loss, "view_kl": loss.detach()}
