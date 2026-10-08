"""Distillation losses: teacher outputs -> student outputs.

All functions take the dicts returned by ReIDNet.forward() (see models/reid_net.py).
Teacher tensors must already be detached / computed under torch.no_grad().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def logit_kd(
    s_logits: torch.Tensor, t_logits: torch.Tensor, temperature: float = 4.0
) -> torch.Tensor:
    """Hinton et al. 2015: KL(teacher || student) on softened logits, scaled by T^2."""
    t = temperature
    return F.kl_div(
        F.log_softmax(s_logits / t, dim=1), F.softmax(t_logits / t, dim=1), reduction="batchmean"
    ) * (t * t)


def similarity_kd(s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
    """Match the in-batch cosine-similarity matrix (relational KD). Dimension-agnostic,
    so it works even when student and teacher feature sizes differ."""
    s = F.normalize(s_feat, dim=1)
    t = F.normalize(t_feat, dim=1)
    return F.mse_loss(s @ s.t(), t @ t.t())


class FeatureKD(nn.Module):
    """L2 between projected student embedding and teacher embedding (FitNets-style, on pooled
    features). The projector is learnable and must be added to the optimizer."""

    def __init__(self, s_dim: int, t_dim: int, normalize: bool = True):
        super().__init__()
        self.proj = nn.Identity() if s_dim == t_dim else nn.Linear(s_dim, t_dim, bias=False)
        self.normalize = normalize

    def forward(self, s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        s = self.proj(s_feat)
        if self.normalize:
            s, t_feat = F.normalize(s, dim=1), F.normalize(t_feat, dim=1)
        return F.mse_loss(s, t_feat)


class DistillLoss(nn.Module):
    """Weighted sum of the KD terms enabled in cfg.kd.losses, e.g.

        kd:
          temperature: 4.0
          losses: {logit: 1.0, feature: 1.0, similarity: 1.0}

    A weight of 0 (or a missing key) disables that term.
    """

    def __init__(self, kd_cfg, s_dim: int, t_dim: int):
        super().__init__()
        w = kd_cfg.get("losses", {}) or {}
        self.w_logit = float(w.get("logit", 0.0))
        self.w_feat = float(w.get("feature", 0.0))
        self.w_sim = float(w.get("similarity", 0.0))
        self.temperature = float(kd_cfg.get("temperature", 4.0))
        self.feat_kd = FeatureKD(s_dim, t_dim) if self.w_feat > 0 else None
        self.feat_key = kd_cfg.get("feat_key", "feat")  # "feat" (pre-BN) or "bn_feat"

    def forward(self, s_out: dict, t_out: dict) -> dict[str, torch.Tensor]:
        terms: dict[str, torch.Tensor] = {}
        if self.w_logit > 0:
            if t_out.get("logits") is None:
                raise RuntimeError(
                    "Teacher logits missing: run the teacher's classifier in train mode"
                )
            terms["kd_logit"] = self.w_logit * logit_kd(
                s_out["logits"], t_out["logits"], self.temperature
            )
        sf, tf = s_out[self.feat_key], t_out[self.feat_key]
        if self.w_feat > 0:
            terms["kd_feat"] = self.w_feat * self.feat_kd(sf, tf)
        if self.w_sim > 0:
            terms["kd_sim"] = self.w_sim * similarity_kd(sf, tf)
        return terms
