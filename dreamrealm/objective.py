import math
import torch
import torch.nn.functional as F


def family_loss(scores, with_edit, tau=1.0, row_weight=0.5, col_weight=0.5):
    """Mean within a family; callers average families, never matrix cells."""
    if not all(math.isfinite(v) for v in (tau, row_weight, col_weight)):
        raise ValueError("Non-finite loss hyperparameter")
    if tau <= 0 or min(row_weight, col_weight) < 0:
        raise ValueError("Require tau > 0 and nonnegative weights")
    if scores.ndim != 2 or not torch.isfinite(scores).all():
        raise ValueError("Scores must be a finite 2D tensor")
    scores = scores.float()
    n, m = scores.shape
    if with_edit:
        if n != m or n < 2:
            raise ValueError("Edited family requires full square N x N, N >= 2")
        labels = torch.arange(n, device=scores.device)
        sft = -scores.diagonal().mean()
        row = F.cross_entropy(scores / tau, labels)
        col = F.cross_entropy(scores.T / tau, labels)
    else:
        if n != 1 or m < 1:
            raise ValueError("Plain family requires 1 x (M+1)")
        sft = -scores[0, 0]
        row = F.cross_entropy(
            scores / tau, torch.zeros(1, dtype=torch.long, device=scores.device)
        )
        col = scores.new_zeros(())
    total = sft + row_weight * row + col_weight * col
    return total, {"sft": sft.detach(), "row": row.detach(), "col": col.detach()}
