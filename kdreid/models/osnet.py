"""OSNet (Zhou et al., ICCV 2019), re-implemented with the same module names as torchreid so the
official ImageNet weights load directly.

Only the convolutional body lives here (conv1 ... conv5 -> B x C x 16 x 8 for a 256x128 input).
torchreid's optional `fc` head (Linear-BN-ReLU, 512-d) maps onto `ReIDNet.embed`; see
`load_osnet_imagenet`.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# From torchreid/models/osnet.py (KaiyangZhou/deep-person-reid).
PRETRAINED_URLS = {
    "osnet_x1_0": "https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY",
    "osnet_x0_75": "https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq",
    "osnet_x0_5": "https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i",
    "osnet_x0_25": "https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs",
}

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


def build_osnet_body(name: str) -> tuple[OSNetBody, int]:
    body = OSNetBody(CHANNELS[name])
    return body, body.out_channels


# ----------------------------------------------------------------------------- weights
def _cache_path(name: str) -> Path:
    torch_home = os.path.expanduser(
        os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
    )
    return Path(torch_home) / "checkpoints" / f"{name}_imagenet.pth"


def resolve_imagenet_weights(name: str, path: str | None = None) -> Path:
    """Use `path` if given (downloading into it if missing — on Colab point it at Drive so the
    download happens once), else the torch cache (same filename torchreid uses)."""
    p = Path(path) if path else _cache_path(name)
    if p.exists():
        return p
    try:
        import gdown
    except ImportError as e:
        raise RuntimeError(
            f"ImageNet weights for {name} not found at {p} and gdown is not installed. "
            "pip install gdown, or set model.pretrained_path to a downloaded file."
        ) from e
    p.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(PRETRAINED_URLS[name], str(p), quiet=False)
    if not p.exists():
        raise RuntimeError(f"Download of {name} ImageNet weights failed ({PRETRAINED_URLS[name]})")
    return p


def load_osnet_imagenet(reid_net: nn.Module, name: str, path: str | None = None) -> list[str]:
    """Load torchreid ImageNet weights into ReIDNet: conv* -> backbone.*, fc.* -> embed.*.

    The 1000-way ImageNet classifier is always skipped. Returns the list of loaded keys.
    """
    src = torch.load(resolve_imagenet_weights(name, path), map_location="cpu", weights_only=False)
    src = src.get("state_dict", src)
    own = reid_net.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for k, v in src.items():
        k = k.removeprefix("module.")
        if k.startswith("classifier."):
            continue
        tgt = "embed." + k[len("fc.") :] if k.startswith("fc.") else "backbone." + k
        if tgt in own and own[tgt].shape == v.shape:
            mapped[tgt] = v
    n_body = sum(k.startswith("backbone.") for k in own)
    n_loaded_body = sum(k.startswith("backbone.") for k in mapped)
    if n_loaded_body != n_body:
        warnings.warn(
            f"OSNet ImageNet weights: only {n_loaded_body}/{n_body} backbone tensors matched",
            stacklevel=2,
        )
    reid_net.load_state_dict(mapped, strict=False)
    return sorted(mapped)
