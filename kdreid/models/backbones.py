"""Backbones. Each returns (module producing a B x C x H x W map, C).

ResNet / MobileNetV3 come from torchvision (ImageNet weights loaded here). OSNet bodies are built
here but their ImageNet weights are loaded by ReIDNet, because torchreid's checkpoint also covers
the embedding head (see models/osnet.py)."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm

from .osnet import CHANNELS as _OSNETS
from .osnet import build_osnet_body

_RESNETS = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.DEFAULT, 512),
    "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.DEFAULT, 512),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.DEFAULT, 2048),
    "resnet101": (tvm.resnet101, tvm.ResNet101_Weights.DEFAULT, 2048),
}
_MOBILENETS = {
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.DEFAULT, 576),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.DEFAULT, 960),
}

AVAILABLE = sorted([*_RESNETS, *_MOBILENETS, *_OSNETS])


def _set_last_stride_1(resnet: nn.Module) -> None:
    """Common ReID trick: keep a 16x8 map (for 256x128 input) instead of 8x4."""
    block = resnet.layer4[0]
    # torchvision Bottleneck puts the stride on conv2; BasicBlock puts it on conv1.
    if isinstance(block, tvm.resnet.Bottleneck):
        block.conv2.stride = (1, 1)
    else:
        block.conv1.stride = (1, 1)
    if block.downsample is not None:
        block.downsample[0].stride = (1, 1)


def build_backbone(name: str, pretrained: bool = True, last_stride: int = 1):
    if name in _RESNETS:
        fn, weights, dim = _RESNETS[name]
        net = fn(weights=weights if pretrained else None)
        if last_stride == 1:
            _set_last_stride_1(net)
        body = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        return body, dim
    if name in _MOBILENETS:
        fn, weights, dim = _MOBILENETS[name]
        net = fn(weights=weights if pretrained else None)
        return net.features, dim
    if name in _OSNETS:
        return build_osnet_body(name)
    raise KeyError(f"Unknown backbone {name!r}; available: {AVAILABLE}")
