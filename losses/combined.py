"""Combined criteria: enable-flag summation for ablations.

Each member runs alone (flag off skips it); the wrapper only sums with
configured weights and reports gradient-conflict cosine for monitoring.
No GradNorm/PCGrad machinery here — monitor first, escalate if needed.
"""
import torch
import torch.nn as nn


def grad_conflict_cosine(grads_a, grads_b) -> float:
    """Cosine between two gradient lists; negative means conflict."""
    dot = sum((a * b).sum() for a, b in zip(grads_a, grads_b) if a is not None and b is not None)
    na = sum((a * a).sum() for a in grads_a if a is not None).sqrt().clamp(min=1e-12)
    nb = sum((b * b).sum() for b in grads_b if b is not None).sqrt().clamp(min=1e-12)
    return float((dot / (na * nb)).detach().cpu())


class CombinedAuxCriterion(nn.Module):
    """L = w_rec * L_rec + w_box * L_box; either member runs alone."""

    def __init__(self, recon=None, box=None, w_rec: float = 1.0,
                 w_box: float = 1.0):
        super().__init__()
        self.recon, self.box = recon, box
        self.w_rec, self.w_box = float(w_rec), float(w_box)

    def forward(self, **kwargs) -> dict:
        """Expect {'recon': ..., 'target': ..., 'masks': ...} and/or box args."""
        out: dict[str, torch.Tensor] = {}
        total = None
        if self.recon is not None and "recon" in kwargs:
            r = self.recon(kwargs["recon"], kwargs["target"], kwargs.get("masks"))
            out.update({f"aux_rec_{k}": v for k, v in r.items()})
            total = self.w_rec * r["loss"] if total is None else total + self.w_rec * r["loss"]
        if self.box is not None and "grid" in kwargs:
            b = self.box(kwargs["grid"], kwargs["box_target"])
            out.update({f"aux_box_{k}": v for k, v in b.items()})
            total = self.w_box * b["loss"] if total is None else total + self.w_box * b["loss"]
        assert total is not None, "CombinedAuxCriterion got no active member"
        out["loss"] = total
        return out


class CombinedHeadCriterion(nn.Module):
    """Weighted sum over any subset of H-losses for the tournament."""

    def __init__(self, members: dict, weights: dict | None = None):
        super().__init__()
        self.members = nn.ModuleDict(members)
        self.weights = dict(weights or {})

    def forward(self, inputs: dict) -> dict:
        """Each member reads inputs[member_name]; missing entries skipped."""
        out: dict[str, torch.Tensor] = {}
        total = None
        for name, loss in self.members.items():
            if name not in inputs:
                continue
            r = loss(**inputs[name]) if isinstance(inputs[name], dict) else loss(inputs[name])
            if isinstance(r, dict):
                out.update({f"{name}_{k}": v for k, v in r.items()})
                val = r["loss"]
            else:
                out[f"{name}"] = r
                val = r
            w = float(self.weights.get(name, 1.0))
            total = w * val if total is None else total + w * val
        assert total is not None, "CombinedHeadCriterion got no active member"
        out["loss"] = total
        return out


class FullCriterion(nn.Module):
    """Full-LeWM criterion: trunk loss plus attached-head losses, one pass.

    Consumes FullLeWMModel outputs. Members activate only when their
    inputs exist: trunk always; recon-aux and H-heads when attached;
    box supervision only when the batch carries 'box_target' (proposal
    wiring, staged later). Always exposes 'pred_loss' for validation.
    """

    def __init__(self, trunk, aux_recon=None, h2=None, h3=None, h4=None,
                 box=None, weights: dict | None = None):
        super().__init__()
        self.trunk = trunk
        self.aux_recon, self.h2 = aux_recon, h2
        self.h3, self.h4, self.box = h3, h4, box
        self.weights = dict(weights or {})

    def _add(self, out, total, key, value, name):
        out[f"{name}_{key}" if key != "loss" else name] = value
        w = float(self.weights.get(name, 1.0))
        return out, (w * value if total is None else total + w * value)

    def forward(self, outputs: dict) -> dict:
        """Combine trunk and head terms into one loss plus 'pred_loss'."""
        out: dict[str, torch.Tensor] = {}
        total = None
        t = self.trunk(outputs["trunk"])
        for key, value in t.items():
            out, total = self._add(out, total, key, value, "trunk")
        out["pred_loss"] = t["pred_loss"]

        if self.aux_recon is not None and "aux_recon" in outputs:
            r = self.aux_recon(outputs["aux_recon"], outputs["window"])
            for key, value in r.items():
                out, total = self._add(out, total, key, value, "aux")
        if self.h2 is not None and "h2" in outputs:
            # Mean-fuse per-level reconstructions to one window estimate.
            recons = list(outputs["h2"]["recon"].values())
            fused = sum(recons) / max(len(recons), 1)
            r = self.h2(fused, outputs["window"])
            for key, value in r.items():
                out, total = self._add(out, total, key, value, "h2")
        if self.h3 is not None and "h3" in outputs:
            r = self.h3(outputs["h3"])
            for key, value in r.items():
                out, total = self._add(out, total, key, value, "h3")
        if self.h4 is not None and "h4" in outputs:
            r = self.h4(outputs["h4"], outputs["h4_centers"],
                        raw_embeds=outputs["h4_raw"])
            for key, value in r.items():
                out, total = self._add(out, total, key, value, "h4")
        if self.box is not None and "box_target" in outputs:
            r = self.box(outputs["aux_boxes"], outputs["box_target"])
            for key, value in r.items():
                out, total = self._add(out, total, key, value, "box")
        assert total is not None, "FullCriterion got no active member"
        out["loss"] = total
        return out
