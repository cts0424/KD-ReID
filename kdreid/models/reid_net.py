"""ReID network: backbone -> GAP -> [embed] -> BNNeck -> classifier (Luo et al., "Bag of Tricks").

`embed` is an optional Linear-BN-ReLU head (OSNet's original 512-d `fc`). It is off by default,
so ResNet models are unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .osnet import CHANNELS as OSNET_NAMES
from .osnet import load_osnet_imagenet


class ReIDNet(nn.Module):
    """forward() returns a dict so distillation can reach every intermediate signal.

    keys:
      feat_map   B x C x h x w   backbone output (for attention / spatial KD)
      feat       B x D           pooled (and embedded) feature before BNNeck (triplet, KD)
      bn_feat    B x D           feature after BNNeck (used for retrieval at test time)
      logits     B x num_classes identity logits (training only; None in eval)
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pretrained: bool = True,
        last_stride: int = 1,
        neck: str = "bnneck",
        embed_dim: int = 0,
        pretrained_path: str | None = None,
    ):
        super().__init__()
        is_osnet = backbone in OSNET_NAMES
        # torchvision loads its own weights; OSNet weights are loaded below (they cover `embed`).
        self.backbone, self.map_dim = build_backbone(
            backbone, pretrained and not is_osnet, last_stride
        )
        self.embed: nn.Module | None = None
        self.feat_dim = self.map_dim
        if embed_dim and embed_dim > 0:
            self.embed = nn.Sequential(
                nn.Linear(self.map_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(inplace=True),
            )
            nn.init.normal_(self.embed[0].weight, 0, 0.01)
            nn.init.zeros_(self.embed[0].bias)
            self.feat_dim = embed_dim
        self.neck_type = neck
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)  # BNNeck: no shift
        nn.init.ones_(self.bottleneck.weight)
        nn.init.zeros_(self.bottleneck.bias)
        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        if is_osnet and pretrained:
            load_osnet_imagenet(self, backbone, pretrained_path)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        fmap = self.backbone(x)
        feat = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        if self.embed is not None:
            feat = self.embed(feat)
        bn_feat = self.bottleneck(feat) if self.neck_type == "bnneck" else feat
        logits = self.classifier(bn_feat) if self.training else None
        return {"feat_map": fmap, "feat": feat, "bn_feat": bn_feat, "logits": logits}

    def backbone_parameters(self):
        """Parameters that came from ImageNet pre-training (frozen during warm-up if asked)."""
        yield from self.backbone.parameters()
        if self.embed is not None:
            yield from self.embed.parameters()


def build_model(model_cfg, num_classes: int) -> ReIDNet:
    return ReIDNet(
        backbone=model_cfg.backbone,
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        last_stride=model_cfg.get("last_stride", 1),
        neck=model_cfg.get("neck", "bnneck"),
        embed_dim=model_cfg.get("embed_dim", 0) or 0,
        pretrained_path=model_cfg.get("pretrained_path"),
    )
