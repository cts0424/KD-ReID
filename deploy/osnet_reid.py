"""Standalone ReID model for deployment — needs only PyTorch (no KD-ReID code).

OSNet (the distilled students) is defined here in full; ResNet (e.g. the R50 teacher, packaged for
comparison) reuses torchvision's ResNet, so `--backend torch` for a ResNet package also needs
torchvision. The ONNX file needs neither.

Same module names as the training code, so the packaged `weights.pt` loads with strict=True.
forward(images) returns the retrieval embedding (BNNeck output, not yet L2-normalised).

    import torch
    from osnet_reid import build_from_metadata
    model = build_from_metadata("metadata.json", "weights.pt").eval()
    feats = model(images)                      # images: B x 3 x 256 x 128, ImageNet-normalised
    feats = torch.nn.functional.normalize(feats, dim=1)   # then cosine similarity = dot product
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = {
    "osnet_x1_0": [64, 256, 384, 512],
    "osnet_x0_75": [48, 192, 288, 384],
    "osnet_x0_5": [32, 128, 192, 256],
    "osnet_x0_25": [16, 64, 96, 128],
}


# ----------------------------------------------------------------------------- layers
class ConvLayer(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """1x1 (linear) + depthwise 3x3, then BN + ReLU."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Unified aggregation gate shared by the four streams (SE-style, sigmoid)."""

    def __init__(self, c: int, reduction: int = 16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, c // reduction, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(c // reduction, c, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        g = self.gate_activation(self.fc2(self.relu(self.fc1(self.global_avgpool(x)))))
        return x * g


class OSBlock(nn.Module):
    """Omni-scale residual block: 4 streams with 1-4 stacked LightConv3x3 (receptive fields
    3, 5, 7, 9), fused by a shared channel gate."""

    def __init__(self, cin: int, cout: int, bottleneck_reduction: int = 4):
        super().__init__()
        mid = cout // bottleneck_reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(2)])
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = (
            self.gate(self.conv2a(x1))
            + self.gate(self.conv2b(x1))
            + self.gate(self.conv2c(x1))
            + self.gate(self.conv2d(x1))
        )
        identity = self.downsample(x) if self.downsample is not None else x
        return F.relu(self.conv3(x2) + identity)


# ----------------------------------------------------------------------------- body
class OSNetBody(nn.Module):
    """Convolutional part of OSNet; forward returns the final feature map."""

    def __init__(self, channels: list[int], layers: tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        c = channels
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(layers[0], c[0], c[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(layers[1], c[1], c[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(layers[2], c[2], c[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(c[3], c[3])
        self.out_channels = c[3]
        self._init_params()

    @staticmethod
    def _make_layer(n: int, cin: int, cout: int, reduce_spatial_size: bool) -> nn.Sequential:
        blocks: list[nn.Module] = [OSBlock(cin, cout)]
        blocks += [OSBlock(cout, cout) for _ in range(1, n)]
        if reduce_spatial_size:
            blocks.append(nn.Sequential(Conv1x1(cout, cout), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*blocks)

    def _init_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        return self.conv5(self.conv4(self.conv3(self.conv2(x))))


class OSNetReID(nn.Module):
    """OSNet body -> global average pooling -> [embed: Linear-BN-ReLU] -> BNNeck."""

    def __init__(self, backbone: str = "osnet_x0_25", embed_dim: int = 512):
        super().__init__()
        self.backbone = OSNetBody(CHANNELS[backbone])
        c = self.backbone.out_channels
        self.embed = None
        if embed_dim:
            self.embed = nn.Sequential(
                nn.Linear(c, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()
            )
            c = embed_dim
        self.bottleneck = nn.BatchNorm1d(c)
        self.feat_dim = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = F.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1)
        if self.embed is not None:
            f = self.embed(f)
        return self.bottleneck(f)


class ResNetReID(nn.Module):
    """torchvision ResNet body (optionally last stride 1) -> global average pooling -> BNNeck.

    Module order matches the training code's nn.Sequential, so weights load with strict=True."""

    def __init__(self, backbone: str = "resnet50", last_stride: int = 1):
        super().__init__()
        import torchvision.models as tvm

        net = getattr(tvm, backbone)(weights=None)
        if last_stride == 1:
            block = net.layer4[0]
            (block.conv2 if hasattr(block, "conv3") else block.conv1).stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
        )
        self.feat_dim = net.fc.in_features
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(F.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1))


def build_model(backbone: str, embed_dim: int = 0, last_stride: int = 1) -> nn.Module:
    if backbone in CHANNELS:
        return OSNetReID(backbone, embed_dim)
    if backbone.startswith("resnet"):
        if embed_dim:
            raise ValueError("ResNet packages have no embedding head")
        return ResNetReID(backbone, last_stride)
    raise ValueError(f"unsupported backbone {backbone!r}")


def build_from_metadata(metadata_path: str, weights_path: str) -> nn.Module:
    with open(metadata_path, encoding="utf-8") as f:
        meta = json.load(f)
    m = meta["model"]
    model = build_model(m["backbone"], m.get("embed_dim", 0), m.get("last_stride", 1))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()
