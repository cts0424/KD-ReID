"""Person re-identification inference for the packaged model. Needs: numpy, pillow, and
onnxruntime (default backend) or torch (--backend torch).

    # 1. check the package works on this machine (compares against bundled reference outputs)
    python reid_infer.py selftest

    # 2. embed every image in a folder -> features.npy (N x D, L2-normalised) + features.json (paths)
    python reid_infer.py extract --images gallery/ --out features

    # 3. rank a gallery folder for one or more query images
    python reid_infer.py search --query q1.jpg q2.jpg --gallery gallery/ --topk 10

Inputs are person crops (any size; resized to 256x128). Similarity = cosine of the L2-normalised
embeddings, in [-1, 1]; higher = more likely the same person. There is no universal threshold:
pick one on labelled pairs from your own cameras.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_meta() -> dict:
    return json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- preprocessing
def preprocess(img: Image.Image, meta: dict) -> np.ndarray:
    """PIL image -> 3 x H x W float32, exactly as in training/evaluation:
    RGB, bilinear resize to (H, W), scale to [0, 1], ImageNet mean/std."""
    p = meta["preprocess"]
    h, w = p["size_hw"]
    img = img.convert("RGB").resize((w, h), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(p["mean"], np.float32)) / np.array(p["std"], np.float32)
    return x.transpose(2, 0, 1)


def list_images(folder: str | Path) -> list[Path]:
    paths = sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        sys.exit(f"no images found under {folder}")
    return paths


# ----------------------------------------------------------------------------- backends
class Embedder:
    """Wraps the ONNX or PyTorch model; returns L2-normalised embeddings."""

    def __init__(self, backend: str = "onnx", device: str = "cpu", flip: bool = True):
        self.meta = load_meta()
        self.flip = flip
        self.backend = backend
        if backend == "onnx":
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if device == "cuda":
                providers.insert(0, "CUDAExecutionProvider")
            self.sess = ort.InferenceSession(str(HERE / "model.onnx"), providers=providers)
        elif backend == "torch":
            import torch

            sys.path.insert(0, str(HERE))
            from osnet_reid import build_from_metadata

            self.torch = torch
            self.device = torch.device(device)
            self.model = build_from_metadata(
                str(HERE / "metadata.json"), str(HERE / "weights.pt")
            ).to(self.device)
        else:
            raise ValueError("backend must be 'onnx' or 'torch'")

    def _raw(self, batch: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            return self.sess.run(None, {"images": batch})[0]
        with self.torch.no_grad():
            t = self.torch.from_numpy(batch).to(self.device)
            return self.model(t).float().cpu().numpy()

    def embed_array(self, batch: np.ndarray) -> np.ndarray:
        """batch: B x 3 x H x W preprocessed float32 -> B x D unit-norm embeddings."""
        f = self._raw(batch)
        if self.flip:  # test-time flip, as used for the reported accuracy
            f = f + self._raw(np.ascontiguousarray(batch[..., ::-1]))
        return f / np.linalg.norm(f, axis=1, keepdims=True).clip(min=1e-12)

    def embed_paths(self, paths: list[Path], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(paths), batch_size):
            arr = np.stack(
                [preprocess(Image.open(p), self.meta) for p in paths[i : i + batch_size]]
            )
            out.append(self.embed_array(arr))
        return np.concatenate(out)


# ----------------------------------------------------------------------------- commands
def cmd_selftest(args) -> None:
    ref = np.load(HERE / "selftest.npz")
    ok = True
    for backend in ["onnx", "torch"] if args.backend == "both" else [args.backend]:
        try:
            emb = Embedder(backend, args.device, flip=False)
        except ImportError as e:
            print(f"[skip] {backend}: {e}")
            continue
        raw = emb._raw(ref["inputs"])
        diff = float(np.abs(raw - ref["outputs"]).max())
        good = diff < 1e-3
        ok &= good
        print(f"[{'ok' if good else 'FAIL'}] {backend}: max |output - reference| = {diff:.2e}")
        x1 = ref["inputs"][:1]
        for _ in range(10):  # warm-up: first calls include graph optimisation / allocation
            emb._raw(x1)
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            emb._raw(x1)
            times.append(time.perf_counter() - t0)
        print(
            f"       {backend}: {np.median(times) * 1000:.1f} ms / image "
            "(batch 1, median of 50 after warm-up, single pass; flip TTA doubles it)"
        )
    sys.exit(0 if ok else 1)


def cmd_extract(args) -> None:
    paths = list_images(args.images)
    feats = Embedder(args.backend, args.device, not args.no_flip).embed_paths(paths, args.batch)
    np.save(f"{args.out}.npy", feats.astype(np.float32))
    Path(f"{args.out}.json").write_text(json.dumps([str(p) for p in paths], indent=1), "utf-8")
    print(f"{len(paths)} images -> {args.out}.npy {feats.shape} + {args.out}.json")


def cmd_search(args) -> None:
    emb = Embedder(args.backend, args.device, not args.no_flip)
    gallery = list_images(args.gallery)
    g = emb.embed_paths(gallery, args.batch)
    q = emb.embed_paths([Path(p) for p in args.query], args.batch)
    sims = q @ g.T
    for qi, qp in enumerate(args.query):
        order = np.argsort(-sims[qi])[: args.topk]
        print(f"\nquery: {qp}")
        for rank, gi in enumerate(order, 1):
            print(f"  {rank:2d}. {sims[qi, gi]:+.4f}  {gallery[gi]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", default="onnx", choices=["onnx", "torch", "both"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="verify outputs against bundled reference")
    e = sub.add_parser("extract", help="embed a folder of images")
    e.add_argument("--images", required=True)
    e.add_argument("--out", default="features")
    s = sub.add_parser("search", help="rank a gallery folder for query images")
    s.add_argument("--query", nargs="+", required=True)
    s.add_argument("--gallery", required=True)
    s.add_argument("--topk", type=int, default=10)
    for p in (e, s):
        p.add_argument("--batch", type=int, default=64)
        p.add_argument("--no-flip", action="store_true", help="single pass (about 2x faster)")
    args = ap.parse_args()
    if args.cmd != "selftest" and args.backend == "both":
        ap.error("--backend both is only for selftest")
    {"selftest": cmd_selftest, "extract": cmd_extract, "search": cmd_search}[args.cmd](args)


if __name__ == "__main__":
    main()
