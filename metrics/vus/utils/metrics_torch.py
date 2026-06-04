import math
import numpy as np
from sklearn import metrics

try:
    import torch
    _HAS_TORCH = True
except ImportError:  # graceful fallback
    _HAS_TORCH = False


class metricor_t:
    def __init__(self, a=1, probability=True, bias="flat"):
        self.a = a
        self.probability = probability
        self.bias = bias

    # ---- vectorized segment extraction -------------------------------
    def range_convers_new(self, label):
        """Return list of (start, end) inclusive index pairs of non-zero runs."""
        x = (np.asarray(label) != 0).astype(np.int8)
        if x.size == 0:
            return []
        d = np.diff(np.concatenate(([0], x, [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1
        return list(zip(starts.tolist(), ends.tolist()))

    # ---- label extension (unchanged logic) ---------------------------
    def extend_postive_range(self, x, window=5):
        label = np.asarray(x).copy().astype(float)
        L = self.range_convers_new(label)
        length = len(label)
        for s, e in L:
            x1 = np.arange(e, min(e + window // 2, length))
            label[x1] += np.sqrt(1 - (x1 - e) / window)
            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / window)
        return np.minimum(1.0, label)

    # ---- optimized volume computation --------------------------------
    def RangeAUC_volume(
        self,
        labels_original,
        score,
        windowSize,
        device=None,
        n_thresholds=250,
        dtype=None,
    ):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for the optimized version.")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            # float32 on GPU is much faster; use float64 on CPU for fidelity
            dtype = torch.float32 if device == "cuda" else torch.float64

        labels_original = np.asarray(labels_original)
        score = np.asarray(score)
        N = score.shape[0]
        P = float(labels_original.sum())

        score_t = torch.as_tensor(score, dtype=dtype, device=device)

        # --- prediction matrix: built ONCE, reused for every window ---
        score_sorted, _ = torch.sort(score_t, descending=True)
        idx = torch.linspace(0, N - 1, n_thresholds, device=device).long()
        thresholds = score_sorted[idx]                      # (T,)
        pred_f = (score_t.unsqueeze(0) >= thresholds.unsqueeze(1)).to(dtype)  # (T,N)

        num_pred = pred_f.sum(dim=1)                        # (T,)
        # padded cumulative sum for O(1) segment lookups
        csum = torch.cumsum(pred_f, dim=1)                  # (T,N)
        zero_col = torch.zeros((n_thresholds, 1), dtype=dtype, device=device)
        pcsum = torch.cat([zero_col, csum], dim=1)          # (T, N+1)

        z1 = torch.zeros(1, dtype=dtype, device=device)
        o1 = torch.ones(1, dtype=dtype, device=device)

        tpr_3d, fpr_3d, prec_3d = [], [], []
        auc_3d, ap_3d = [], []
        window_3d = np.arange(0, windowSize + 1, 1)

        for window in window_3d:
            labels = self.extend_postive_range(labels_original, window)
            labels_t = torch.as_tensor(labels, dtype=dtype, device=device)
            L = self.range_convers_new(labels)
            nL = max(len(L), 1)

            starts = torch.tensor([s for s, _ in L], device=device, dtype=torch.long)
            ends = torch.tensor([e for _, e in L], device=device, dtype=torch.long)

            P_new = (P + float(labels.sum())) / 2.0
            N_new = N - P_new

            TP = pred_f @ labels_t                          # (T,)
            recall = torch.clamp(TP / P_new, max=1.0)

            seg_sums = pcsum[:, ends + 1] - pcsum[:, starts]  # (T, nL)
            existence = (seg_sums > 0).to(dtype).sum(dim=1)
            existence_ratio = existence / nL

            TPR = recall * existence_ratio
            FP = num_pred - TP
            FPR = FP / N_new
            Precision = TP / num_pred                       # num_pred >= 1 always

            tpr = torch.cat([z1, TPR, o1])                  # (T+2,)
            fpr = torch.cat([z1, FPR, o1])
            prec = torch.cat([o1, Precision])               # (T+1,)

            width = fpr[1:] - fpr[:-1]
            height = (tpr[1:] + tpr[:-1]) / 2
            auc_3d.append(torch.sum(width * height).item())

            width_pr = tpr[1:-1] - tpr[:-2]
            height_pr = (prec[1:] + prec[:-1]) / 2
            ap_3d.append(torch.sum(width_pr * height_pr).item())

            tpr_3d.append(tpr.cpu().numpy())
            fpr_3d.append(fpr.cpu().numpy())
            prec_3d.append(prec.cpu().numpy())

        avg_auc = sum(auc_3d) / len(window_3d)
        avg_ap = sum(ap_3d) / len(window_3d)
        return tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc, avg_ap


def generate_curve(label, score, slidingWindow, device=None):
    tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = (
        metricor_t().RangeAUC_volume(
            labels_original=label,
            score=score,
            windowSize=1 * slidingWindow,
            device=device,
        )
    )

    X = np.array(tpr_3d).reshape(1, -1).ravel()
    X_ap = np.array(tpr_3d)[:, :-1].reshape(1, -1).ravel()
    Y = np.array(fpr_3d).reshape(1, -1).ravel()
    W = np.array(prec_3d).reshape(1, -1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0]) - 1)

    return Y, Z, X, X_ap, W, Z_ap, avg_auc_3d, avg_ap_3d