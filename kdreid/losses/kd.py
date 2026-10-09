"""Distillation losses: teacher outputs -> student outputs.

All functions take the dicts returned by ReIDNet.forward() (see models/reid_net.py).
Teacher tensors must already be detached / computed under torch.no_grad().

| cfg key     | method                                              | needs              |
|-------------|-----------------------------------------------------|--------------------|
| logit       | Hinton KD, KL on softened logits                    | logits             |
| dkd         | Decoupled KD (Zhao et al. CVPR'22), TCKD + NCKD      | logits + labels    |
| feature     | projected L2 on embeddings (FitNets-style)          | feat               |
| similarity  | MSE between in-batch cosine-similarity matrices     | feat               |
| simdist     | KL between per-row similarity distributions         | feat               |
| rkd         | Relational KD (Park et al. CVPR'19), distance+angle | feat               |
| attention   | attention transfer (Zagoruyko & Komodakis ICLR'17)  | feat_map           |

The relation-based terms (similarity, simdist, rkd, attention) are independent of channel /
embedding size, which is what makes them usable across very different architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- logit-based


def logit_kd(
    s_logits: torch.Tensor, t_logits: torch.Tensor, temperature: float = 4.0
) -> torch.Tensor:
    """Hinton et al. 2015: KL(teacher || student) on softened logits, scaled by T^2."""
    t = temperature
    return F.kl_div(
        F.log_softmax(s_logits / t, dim=1), F.softmax(t_logits / t, dim=1), reduction="batchmean"
    ) * (t * t)


def dkd_loss(
    s_logits: torch.Tensor,
    t_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 1.0,
    beta: float = 8.0,
) -> torch.Tensor:
    """Decoupled KD: alpha * TCKD (target vs. rest, binary) + beta * NCKD (among non-targets).

    Classic KD couples NCKD with the teacher's target confidence, so a confident teacher (or one
    trained with label smoothing) transfers little "dark knowledge"; DKD weights it explicitly.
    """
    t = temperature
    s, tt = s_logits.float() / t, t_logits.float() / t
    gt = F.one_hot(targets, s.size(1)).bool()

    p_s, p_t = F.softmax(s, dim=1), F.softmax(tt, dim=1)
    b_s = torch.stack([p_s[gt], (p_s * ~gt).sum(1)], dim=1).clamp_min(1e-8)
    b_t = torch.stack([p_t[gt], (p_t * ~gt).sum(1)], dim=1).clamp_min(1e-8)
    tckd = F.kl_div(b_s.log(), b_t, reduction="batchmean")

    # distributions over non-target classes only (mask the target logit away)
    big = torch.finfo(s.dtype).max / 4
    nckd = F.kl_div(
        F.log_softmax(s.masked_fill(gt, -big), dim=1),
        F.softmax(tt.masked_fill(gt, -big), dim=1),
        reduction="batchmean",
    )
    return (alpha * tckd + beta * nckd) * (t * t)


# ----------------------------------------------------------------------------- feature-based


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


# ----------------------------------------------------------------------------- relation-based


def similarity_kd(s_feat: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
    """Match the in-batch cosine-similarity matrix (relational KD). Dimension-agnostic,
    so it works even when student and teacher feature sizes differ."""
    s = F.normalize(s_feat, dim=1)
    t = F.normalize(t_feat, dim=1)
    return F.mse_loss(s @ s.t(), t @ t.t())


def simdist_kd(s_feat: torch.Tensor, t_feat: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """For each anchor, KL between the teacher's and student's softmax over cosine similarities
    to the other samples in the batch. Transfers *who is closer than whom* (the ranking that
    retrieval is evaluated on) rather than absolute similarity values."""
    s = F.normalize(s_feat.float(), dim=1)
    t = F.normalize(t_feat.float(), dim=1)
    n = s.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=s.device)
    big = torch.finfo(s.dtype).max / 4
    s_sim = (s @ s.t() / tau).masked_fill(eye, -big)
    t_sim = (t @ t.t() / tau).masked_fill(eye, -big)
    return F.kl_div(F.log_softmax(s_sim, 1), F.softmax(t_sim, 1), reduction="batchmean")


def _pdist(e: torch.Tensor) -> torch.Tensor:
    sq = e.pow(2).sum(1)
    d = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2 * e @ e.t()).clamp_min(1e-12).sqrt()
    return d * (1 - torch.eye(e.size(0), device=e.device))


def rkd_loss(
    s_feat: torch.Tensor, t_feat: torch.Tensor, w_dist: float = 1.0, w_angle: float = 2.0
) -> torch.Tensor:
    """Relational KD: match pairwise distances (normalised by their mean) and triplet angles."""
    s, t = s_feat.float(), t_feat.float()
    loss = s.new_zeros(())
    if w_dist > 0:
        td = _pdist(t)
        td = td / td[td > 0].mean()
        sd = _pdist(s)
        sd = sd / sd[sd > 0].mean()
        loss = loss + w_dist * F.smooth_l1_loss(sd, td)
    if w_angle > 0:

        def angles(e: torch.Tensor) -> torch.Tensor:
            v = F.normalize(e.unsqueeze(0) - e.unsqueeze(1), p=2, dim=2)  # n x n x d
            return torch.bmm(v, v.transpose(1, 2)).flatten()

        loss = loss + w_angle * F.smooth_l1_loss(angles(s), angles(t))
    return loss


def attention_map(fmap: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Spatial attention: channel-mean of |activation|^p, flattened and L2-normalised."""
    return F.normalize(fmap.float().abs().pow(p).mean(1).flatten(1), dim=1)


def attention_kd(s_map: torch.Tensor, t_map: torch.Tensor, p: int = 2) -> torch.Tensor:
    """Attention transfer. Channel counts may differ; spatial sizes are matched by resizing the
    student map to the teacher's if needed (both are 16x8 for R50 / OSNet at 256x128)."""
    if s_map.shape[-2:] != t_map.shape[-2:]:
        s_map = F.interpolate(s_map, size=t_map.shape[-2:], mode="bilinear", align_corners=False)
    return (attention_map(s_map, p) - attention_map(t_map, p)).pow(2).sum(1).mean()


# ----------------------------------------------------------------------------- combined


class DistillLoss(nn.Module):
    """Weighted sum of the KD terms enabled in cfg.kd.losses, e.g.

        kd:
          temperature: 4.0
          losses: {logit: 1.0, similarity: 1.0, attention: 1.0}
          dkd: {alpha: 1.0, beta: 8.0}        # used when losses.dkd > 0
          rkd: {distance: 1.0, angle: 2.0}    # used when losses.rkd > 0
          simdist_tau: 0.1                    # used when losses.simdist > 0

    A weight of 0 (or a missing key) disables that term.
    """

    KEYS = ("logit", "dkd", "feature", "similarity", "simdist", "rkd", "attention")

    def __init__(self, kd_cfg, s_dim: int, t_dim: int):
        super().__init__()
        losses = kd_cfg.get("losses", {}) or {}
        unknown = set(losses) - set(self.KEYS)
        if unknown:
            raise KeyError(f"Unknown kd.losses keys {sorted(unknown)}; allowed: {self.KEYS}")
        self.w = {k: float(losses.get(k, 0.0) or 0.0) for k in self.KEYS}
        self.temperature = float(kd_cfg.get("temperature", 4.0))
        self.feat_key = kd_cfg.get("feat_key", "feat")  # "feat" (pre-BN) or "bn_feat"
        dkd = kd_cfg.get("dkd", {}) or {}
        self.dkd_alpha, self.dkd_beta = float(dkd.get("alpha", 1.0)), float(dkd.get("beta", 8.0))
        rkd = kd_cfg.get("rkd", {}) or {}
        self.rkd_dist, self.rkd_angle = (
            float(rkd.get("distance", 1.0)),
            float(rkd.get("angle", 2.0)),
        )
        self.simdist_tau = float(kd_cfg.get("simdist_tau", 0.1))
        self.at_p = int(kd_cfg.get("attention_p", 2))
        self.feat_kd = FeatureKD(s_dim, t_dim) if self.w["feature"] > 0 else None

    @property
    def needs_logits(self) -> bool:
        return self.w["logit"] > 0 or self.w["dkd"] > 0

    def forward(
        self, s_out: dict, t_out: dict, targets: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        w, terms = self.w, {}
        if self.needs_logits and t_out.get("logits") is None:
            raise RuntimeError("Teacher logits missing: run the teacher's classifier")
        if w["logit"] > 0:
            terms["kd_logit"] = w["logit"] * logit_kd(
                s_out["logits"], t_out["logits"], self.temperature
            )
        if w["dkd"] > 0:
            if targets is None:
                raise RuntimeError("DKD needs the identity labels (targets)")
            terms["kd_dkd"] = w["dkd"] * dkd_loss(
                s_out["logits"],
                t_out["logits"],
                targets,
                self.temperature,
                self.dkd_alpha,
                self.dkd_beta,
            )
        sf, tf = s_out[self.feat_key], t_out[self.feat_key]
        if w["feature"] > 0:
            terms["kd_feat"] = w["feature"] * self.feat_kd(sf, tf)
        if w["similarity"] > 0:
            terms["kd_sim"] = w["similarity"] * similarity_kd(sf, tf)
        if w["simdist"] > 0:
            terms["kd_simdist"] = w["simdist"] * simdist_kd(sf, tf, self.simdist_tau)
        if w["rkd"] > 0:
            terms["kd_rkd"] = w["rkd"] * rkd_loss(sf, tf, self.rkd_dist, self.rkd_angle)
        if w["attention"] > 0:
            terms["kd_at"] = w["attention"] * attention_kd(
                s_out["feat_map"], t_out["feat_map"], self.at_p
            )
        return terms
