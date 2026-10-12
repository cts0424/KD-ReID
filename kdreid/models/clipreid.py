"""CLIP-ReID image branch as a frozen distillation teacher (Li et al., AAAI 2023).

Only the image encoder + BNNecks + identity classifiers are rebuilt; the text branch / prompt
learner are training-time only and are skipped when loading. Module names follow the official
code (github.com/Syliz517/CLIP-ReID, MIT licence; ViT blocks from OpenAI CLIP, MIT licence), so the
released checkpoints load directly.

Output follows ReIDNet's dict contract:
  feat / bn_feat  B x 1280   concat(CLS token, projected CLS token) *before* BN = the official
                             test feature (TEST.NECK_FEAT 'before'), so evaluating this model with
                             our evaluator reproduces the published retrieval numbers
  logits          B x C      identity logits from the BNNeck features; always returned (also in
                             eval mode) because the teacher is used frozen. `logit_head` picks the
                             768-d head, the 512-d projected head, or their mean (default)
  feat_map        B x 768 x h x w   patch tokens of the last layer (for spatial KD)

Inputs are ImageNet-normalised like every other model here; the CLIP normalisation
(mean = std = 0.5) is applied inside, so the student's data pipeline is unchanged.
Supported: ViT-B/16 CLIP-ReID with or without overlapping patches (stride is inferred from the
checkpoint). Not supported: SIE camera embeddings (`cv_embed`), which need camera ids at
inference time.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD

CLIP_MEAN = (0.5, 0.5, 0.5)
CLIP_STD = (0.5, 0.5, 0.5)
ARCHS = {"clipreid_vit_b16": dict(width=768, layers=12, heads=12, patch=16, output_dim=512)}


# ----------------------------------------------------------------------------- CLIP ViT blocks
class LayerNorm(nn.LayerNorm):
    """LayerNorm computed in fp32 (as in CLIP), returned in the input dtype."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )


class VisionTransformer(nn.Module):
    """CLIP ViT with a rectangular token grid and optional overlapping patches (CLIP-ReID)."""

    def __init__(self, h: int, w: int, patch: int, stride: int, width: int, layers: int,
                 heads: int, output_dim: int):  # fmt: skip
        super().__init__()
        self.h, self.w = h, w
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch, stride=stride, bias=False)
        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h * w + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x).flatten(2).transpose(1, 2)  # B x N x width
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).permute(1, 0, 2)  # NLD -> LND
        x = self.transformer.resblocks(x).permute(1, 0, 2)
        x = self.ln_post(x)
        return x, x @ self.proj  # tokens (B x 1+N x width), projected (B x 1+N x output_dim)


# ----------------------------------------------------------------------------- ReID wrapper
class CLIPReIDNet(nn.Module):
    def __init__(self, num_classes: int, arch: str = "clipreid_vit_b16",
                 size: tuple[int, int] = (256, 128), stride: int = 16, logit_head: str = "mean"):  # fmt: skip
        super().__init__()
        a = ARCHS[arch]
        h = (size[0] - a["patch"]) // stride + 1
        w = (size[1] - a["patch"]) // stride + 1
        self.image_encoder = VisionTransformer(
            h, w, a["patch"], stride, a["width"], a["layers"], a["heads"], a["output_dim"]
        )
        self.in_planes, self.in_planes_proj = a["width"], a["output_dim"]
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier_proj = nn.Linear(self.in_planes_proj, num_classes, bias=False)
        if logit_head not in ("mean", "cls", "proj"):
            raise ValueError("logit_head must be 'mean', 'cls' or 'proj'")
        self.logit_head = logit_head
        self.feat_dim = self.in_planes + self.in_planes_proj
        # ImageNet-normalised input -> CLIP-normalised input: x_clip = x * a + b
        im_m, im_s = torch.tensor(IMAGENET_MEAN), torch.tensor(IMAGENET_STD)
        cm, cs = torch.tensor(CLIP_MEAN), torch.tensor(CLIP_STD)
        self.register_buffer("in_scale", (im_s / cs).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_shift", ((im_m - cm) / cs).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, proj = self.image_encoder(x * self.in_scale + self.in_shift)
        f, f_proj = tokens[:, 0], proj[:, 0]
        b, b_proj = self.bottleneck(f), self.bottleneck_proj(f_proj)
        l_cls, l_proj = self.classifier(b), self.classifier_proj(b_proj)
        logits = {"cls": l_cls, "proj": l_proj, "mean": (l_cls + l_proj) / 2}[self.logit_head]
        feat = torch.cat([f, f_proj], dim=1)
        enc = self.image_encoder
        fmap = tokens[:, 1:].transpose(1, 2).reshape(x.shape[0], -1, enc.h, enc.w)
        return {"feat_map": fmap, "feat": feat, "bn_feat": feat, "logits": logits}

    # -------------------------------------------------------------- official checkpoint loading
    KEEP = ("image_encoder.", "bottleneck.", "bottleneck_proj.", "classifier.", "classifier_proj.")

    @classmethod
    def checkpoint_state(cls, path: str) -> dict[str, torch.Tensor]:
        """Image-branch tensors of an official CLIP-ReID checkpoint (or of a model saved here)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        if "cv_embed" in state:
            raise ValueError(
                f"{path} uses SIE camera embeddings (cv_embed), which need camera ids at inference "
                "and are not supported; use the ViT-CLIP-ReID checkpoint without SIE."
            )
        return {k: v.float() if v.is_floating_point() else v for k, v in state.items()
                if k.startswith(cls.KEEP)}  # fmt: skip

    def load_checkpoint(self, path: str) -> None:
        state = self.checkpoint_state(path)
        n_cls = state["classifier.weight"].shape[0]
        if n_cls != self.classifier.out_features:
            raise ValueError(
                f"checkpoint has {n_cls} identities but this dataset has "
                f"{self.classifier.out_features}: logit KD needs a teacher trained on the same set"
            )
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise RuntimeError(
                f"CLIP-ReID load mismatch: missing {missing}, unexpected {unexpected}"
            )


def stride_from_checkpoint(path: str, size=(256, 128), patch: int = 16) -> int:
    """Infer the patch stride (16, or 12 for overlapping patches) from the position embedding."""
    n = CLIPReIDNet.checkpoint_state(path)["image_encoder.positional_embedding"].shape[0] - 1
    for s in (16, 14, 12, 10, 8):
        if ((size[0] - patch) // s + 1) * ((size[1] - patch) // s + 1) == n:
            return s
    raise ValueError(f"cannot infer patch stride from {n} position tokens at input {size}")
