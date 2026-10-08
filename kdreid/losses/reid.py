"""Standard ReID supervision: label-smoothed CE + batch-hard triplet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=self.epsilon)


def pairwise_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xx = x.pow(2).sum(1, keepdim=True)
    yy = y.pow(2).sum(1, keepdim=True).t()
    dist = xx + yy - 2.0 * x @ y.t()
    return dist.clamp(min=1e-12).sqrt()


class TripletLoss(nn.Module):
    """Batch-hard triplet loss (Hermans et al. 2017). margin=None -> soft-margin variant."""

    def __init__(self, margin: float | None = 0.3, normalize: bool = False):
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def forward(self, feats: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            feats = F.normalize(feats, dim=1)
        dist = pairwise_euclidean(feats, feats)
        same = targets.unsqueeze(0) == targets.unsqueeze(1)
        # hardest positive: max distance among same id; hardest negative: min among different id
        d_ap = (dist * same.float()).max(dim=1).values
        d_an = dist.masked_fill(same, float("inf")).min(dim=1).values
        y = torch.ones_like(d_an)
        if self.margin is None:
            return F.soft_margin_loss(d_an - d_ap, y)
        return F.margin_ranking_loss(d_an, d_ap, y, margin=self.margin)
