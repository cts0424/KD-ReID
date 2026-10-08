"""Baseline ReID network: backbone -> GAP -> BNNeck -> classifier (Luo et al., "Bag of Tricks")."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone


def _init_kaiming(m: nn.Module) -> None:
    if isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class ReIDNet(nn.Module):
    """forward() returns a dict so distillation can reach every intermediate signal.

    keys:
      feat_map   B x C x h x w   backbone output (for feature / attention KD)
      feat       B x C           pooled feature before BNNeck (used by triplet loss)
      bn_feat    B x C           feature after BNNeck (used for retrieval at test time)
      logits     B x num_classes identity logits (training only; None in eval)
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        last_stride: int = 1,
        neck: str = "bnneck",
    ):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(backbone, pretrained, last_stride)
        self.neck_type = neck
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)  # BNNeck: no shift
        self.bottleneck.apply(_init_kaiming)
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        fmap = self.backbone(x)
        feat = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        bn_feat = self.bottleneck(feat) if self.neck_type == "bnneck" else feat
        logits = self.classifier(bn_feat) if self.training else None
        return {"feat_map": fmap, "feat": feat, "bn_feat": bn_feat, "logits": logits}


def build_model(model_cfg, num_classes: int) -> ReIDNet:
    return ReIDNet(
        backbone=model_cfg.backbone,
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        last_stride=model_cfg.get("last_stride", 1),
        neck=model_cfg.get("neck", "bnneck"),
    )
