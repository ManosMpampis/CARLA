"""YOLO-1D interval loss: objectness + box regression on synthetic boxes.

Default box term is (1 - 1D-IoU) + L1(center, length); the aspect-aware
CIoU term and DFL stay optional flags for the tournament variant.
Trained on SubAnomaly synthetic boxes only — never on test labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def interval_iou_1d(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """1D IoU between (..., 2) center/length boxes in normalized units."""
    pc, pl = pred[..., 0], pred[..., 1].clamp(min=1e-6)
    tc, tl = tgt[..., 0], tgt[..., 1].clamp(min=1e-6)
    inter = (torch.minimum(pc + pl / 2, tc + tl / 2)
             - torch.maximum(pc - pl / 2, tc - tl / 2)).clamp(min=0)
    union = pl + tl - inter
    return inter / union.clamp(min=1e-9)


class BoxLoss(nn.Module):
    """Objectness (focal BCE) + box regression for (B, G, 3) grids."""

    def __init__(self, lambda_box: float = 1.0, lambda_obj: float = 1.0,
                 focal_gamma: float = 2.0, use_ciou: bool = False,
                 neg_pos_ratio: int = 4):
        super().__init__()
        self.lambda_box, self.lambda_obj = float(lambda_box), float(lambda_obj)
        self.focal_gamma = float(focal_gamma)
        self.use_ciou = bool(use_ciou)
        self.neg_pos_ratio = int(neg_pos_ratio)

    def _hard_negative_mask(self, bce: torch.Tensor, obj: torch.Tensor,
                            keep: torch.Tensor) -> torch.Tensor:
        """Keep all positives + hardest negatives at 1:`neg_pos_ratio`."""
        pos = (obj > 0.5) & (keep > 0.5)
        neg = (~pos) & (keep > 0.5)
        n_pos = pos.sum().clamp(min=1).item()
        n_keep = min(neg.sum().item(), self.neg_pos_ratio * int(n_pos))
        if n_keep <= 0:
            return pos.float()
        neg_scores = (bce * neg.float()).flatten()
        thresh = neg_scores.kthvalue(max(len(neg_scores) - n_keep + 1, 1)).values
        hard = neg & (bce >= thresh)
        return (pos | hard).float()

    def forward(self, grid: torch.Tensor, target: dict) -> dict:
        """Score grid (B, G, 3+) against {'obj', 'boxes', 'mask'} targets."""
        obj_logit, boxes = grid[..., 0], grid[..., 1:3]
        obj = target["obj"].float()
        bce = F.binary_cross_entropy_with_logits(obj_logit, obj, reduction="none")
        # Focal down-weighting for the background-heavy 1D grid.
        focal = (torch.sigmoid(obj_logit).detach() - obj).abs().pow(self.focal_gamma)
        keep = target.get("mask", torch.ones_like(obj))
        mined = self._hard_negative_mask(bce.detach(), obj, keep)
        obj_loss = (bce * focal * mined).sum() / mined.sum().clamp(min=1.0)
        iou = interval_iou_1d(boxes, target["boxes"])
        box_loss = ((1 - iou) * obj * keep).sum() / (obj * keep).sum().clamp(min=1.0)
        l1 = ((boxes - target["boxes"]).abs().mean(-1) * obj * keep).sum()
        l1 = l1 / (obj * keep).sum().clamp(min=1.0)
        if self.use_ciou:  # aspect-aware center-distance penalty (variant)
            dc = ((boxes[..., 0] - target["boxes"][..., 0]) ** 2 * obj * keep).sum()
            dc = dc / (obj * keep).sum().clamp(min=1.0)
            box_loss = box_loss + 0.5 * dc
        total = self.lambda_obj * obj_loss + self.lambda_box * (box_loss + 0.5 * l1)
        return {"obj": obj_loss, "box": box_loss, "l1": l1, "loss": total}
