"""CLIP-ReID image branch as a frozen distillation teacher (Li et al., AAAI 2023).

The image encoder + BNNecks + identity classifiers are always rebuilt. With
`logit_head: text` the learned identity prompts (prompt learner) and the CLIP text encoder are
also loaded and turned into one text prototype per identity: CLIP-ReID's identity knowledge is
trained through these prototypes (the stage-2 i2t loss), while its linear heads stay nearly flat
(see tools/check_teacher.py). Module names follow the official
code (github.com/Syliz517/CLIP-ReID, MIT licence; ViT blocks from OpenAI CLIP, MIT licence), so the
released checkpoints load directly.

Output follows ReIDNet's dict contract:
  feat / bn_feat  B x 1280   concat(CLS token, projected CLS token) *before* BN = the official
                             test feature (TEST.NECK_FEAT 'before'), so evaluating this model with
                             our evaluator reproduces the published retrieval numbers
  logits          B x C      identity logits from the BNNeck features; always returned (also in
                             eval mode) because the teacher is used frozen. `logit_head` picks the
                             768-d head, the 512-d projected head, their mean (default), or
                             `text`: projected CLS feature . text prototype, i.e. exactly the
                             image-to-text logits CLIP-ReID's stage 2 is trained with
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
    """LayerNorm computed in fp32 (as in CLIP), returned in the input dtype. The affine
    parameters are cast too, so a model converted with .half() (fp16 benchmark) still works."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = None if self.weight is None else self.weight.float()
        b = None if self.bias is None else self.bias.float()
        y = nn.functional.layer_norm(x.float(), self.normalized_shape, w, b, self.eps)
        return y.type(x.dtype)


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

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ln_1(x)
        x = x + self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)[0]
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


class TextBranch(nn.Module):
    """CLIP-ReID prompt learner + CLIP text encoder, used only to compute identity prototypes.

    The official prompt is "A photo of a X X X X person." with the four X tokens replaced by a
    learned per-identity context (`cls_ctx`); `token_prefix` / `token_suffix` are the frozen token
    embeddings around them and are stored in the checkpoint, so no tokenizer is needed. The text
    feature is read at the end-of-text token: SOT a photo of a | X X X X | person . EOT -> index
    prefix_len + n_ctx + 2 = 11 (override with `text_eot`; check_teacher.py sweeps it)."""

    def __init__(self, n_cls: int, n_ctx: int, prefix_len: int, context: int, width: int,
                 layers: int, heads: int, embed_dim: int):  # fmt: skip
        super().__init__()
        self.cls_ctx = nn.Parameter(torch.zeros(n_cls, n_ctx, width))
        self.register_buffer("token_prefix", torch.zeros(1, prefix_len, width))
        self.register_buffer("token_suffix", torch.zeros(1, context - prefix_len - n_ctx, width))
        self.positional_embedding = nn.Parameter(torch.zeros(context, width))
        self.transformer = Transformer(width, layers, heads)
        self.ln_final = LayerNorm(width)
        self.text_projection = nn.Parameter(torch.zeros(width, embed_dim))
        mask = torch.full((context, context), float("-inf")).triu(1)  # causal, as in CLIP
        self.register_buffer("attn_mask", mask, persistent=False)
        self.default_eot = prefix_len + n_ctx + 2

    @classmethod
    def from_state(cls, state: dict[str, torch.Tensor]) -> TextBranch:
        n_cls, n_ctx, width = state["cls_ctx"].shape
        context = state["positional_embedding"].shape[0]
        layers = 1 + max(int(k.split(".")[2]) for k in state if k.startswith("transformer."))
        m = cls(n_cls, n_ctx, state["token_prefix"].shape[1], context, width, layers,
                width // 64, state["text_projection"].shape[1])  # fmt: skip
        m.load_state_dict(state, strict=True)
        return m

    @torch.no_grad()
    def prototypes(self, eot: int | None = None, chunk: int = 128) -> torch.Tensor:
        """One text feature per identity (n_cls x embed_dim), computed in fp32."""
        eot = self.default_eot if eot is None else eot
        dev = self.cls_ctx.device
        out = []
        with torch.autocast(device_type=dev.type, enabled=False):
            for i in range(0, self.cls_ctx.shape[0], chunk):
                ctx = self.cls_ctx[i : i + chunk].float()
                b = ctx.shape[0]
                x = torch.cat(
                    [self.token_prefix.expand(b, -1, -1), ctx, self.token_suffix.expand(b, -1, -1)],
                    dim=1,
                )
                x = (x + self.positional_embedding).permute(1, 0, 2)  # NLD -> LND
                for blk in self.transformer.resblocks:
                    x = blk(x, attn_mask=self.attn_mask)
                x = self.ln_final(x.permute(1, 0, 2))
                out.append(x[:, eot] @ self.text_projection)
        return torch.cat(out)


# ----------------------------------------------------------------------------- ReID wrapper
class CLIPReIDNet(nn.Module):
    def __init__(self, num_classes: int, arch: str = "clipreid_vit_b16",
                 size: tuple[int, int] = (256, 128), stride: int = 16, logit_head: str = "mean",
                 text_eot: int | None = None):  # fmt: skip
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
        if logit_head not in ("mean", "cls", "proj", "text"):
            raise ValueError("logit_head must be 'mean', 'cls', 'proj' or 'text'")
        self.logit_head = logit_head
        self.text: TextBranch | None = None  # built from the checkpoint for logit_head='text'
        self.text_eot = text_eot
        self._protos: torch.Tensor | None = None
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
        if self.logit_head == "text":
            with torch.autocast(device_type=x.device.type, enabled=False):
                logits = f_proj.float() @ self.text_prototypes().t()
        else:
            l_cls, l_proj = self.classifier(b), self.classifier_proj(b_proj)
            logits = {"cls": l_cls, "proj": l_proj, "mean": (l_cls + l_proj) / 2}[self.logit_head]
        feat = torch.cat([f, f_proj], dim=1)
        enc = self.image_encoder
        fmap = tokens[:, 1:].transpose(1, 2).reshape(x.shape[0], -1, enc.h, enc.w)
        return {"feat_map": fmap, "feat": feat, "bn_feat": feat, "logits": logits}

    def text_prototypes(self) -> torch.Tensor:
        """Identity text prototypes, computed once on the model's device and cached."""
        if self.text is None:
            raise RuntimeError("logit_head='text' needs a checkpoint with the text branch loaded")
        dev = self.text.cls_ctx.device
        if self._protos is None or self._protos.device != dev:
            self._protos = self.text.prototypes(self.text_eot)
        return self._protos

    def set_text_eot(self, eot: int | None) -> None:
        """Change the end-of-text index (check_teacher.py sweeps it) and drop the cache."""
        self.text_eot, self._protos = eot, None

    # -------------------------------------------------------------- official checkpoint loading
    KEEP = ("image_encoder.", "bottleneck.", "bottleneck_proj.", "classifier.", "classifier_proj.")

    @staticmethod
    def text_state(path: str) -> dict[str, torch.Tensor]:
        """Prompt-learner + text-encoder tensors of an official checkpoint, renamed for
        TextBranch (prompt_learner.cls_ctx -> cls_ctx, text_encoder.ln_final.* -> ln_final.*)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        out = {}
        for k, v in state.items():
            k = k.replace("module.", "", 1)
            for pre in ("prompt_learner.", "text_encoder."):
                if k.startswith(pre):
                    out[k[len(pre) :]] = v.float() if v.is_floating_point() else v
        need = ("cls_ctx", "token_prefix", "token_suffix", "positional_embedding",
                "ln_final.weight", "text_projection")  # fmt: skip
        missing = [k for k in need if k not in out]
        if missing:
            raise ValueError(f"{path} has no usable CLIP-ReID text branch (missing {missing})")
        return out

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
        if self.logit_head == "text":
            text = TextBranch.from_state(self.text_state(path))
            if text.cls_ctx.shape[0] != n_cls:
                raise ValueError(f"text branch has {text.cls_ctx.shape[0]} identities, not {n_cls}")
            if text.text_projection.shape[1] != self.in_planes_proj:
                raise ValueError("text embedding size does not match the image projection")
            self.text = text.to(self.classifier.weight.device)
            self._protos = None


def stride_from_checkpoint(path: str, size=(256, 128), patch: int = 16) -> int:
    """Infer the patch stride (16, or 12 for overlapping patches) from the position embedding."""
    n = CLIPReIDNet.checkpoint_state(path)["image_encoder.positional_embedding"].shape[0] - 1
    for s in (16, 14, 12, 10, 8):
        if ((size[0] - patch) // s + 1) * ((size[1] - patch) // s + 1) == n:
            return s
    raise ValueError(f"cannot infer patch stride from {n} position tokens at input {size}")
