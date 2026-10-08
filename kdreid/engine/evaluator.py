"""Market-1501 style evaluation: CMC rank-k and mAP.

Protocol: for each query, gallery images with the same pid AND same camera are discarded.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def extract_features(model, loader, device, flip: bool = True, feat_key: str = "bn_feat"):
    model.eval()
    feats, pids, cams = [], [], []
    for imgs, pid, cam in tqdm(loader, desc="extract", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        f = model(imgs)[feat_key]
        if flip:  # test-time flip augmentation
            f = f + model(torch.flip(imgs, dims=[3]))[feat_key]
        feats.append(f.float().cpu())
        pids.append(pid)
        cams.append(cam)
    return torch.cat(feats), torch.cat(pids).numpy(), torch.cat(cams).numpy()


def eval_market(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_cams: np.ndarray,
    g_cams: np.ndarray,
    max_rank: int = 50,
) -> tuple[np.ndarray, float]:
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)

    all_cmc, all_ap = [], []
    for i in range(num_q):
        order = indices[i]
        keep = ~((g_pids[order] == q_pids[i]) & (g_cams[order] == q_cams[i]))
        raw = matches[i][keep]
        if not raw.any():  # query identity absent from gallery
            continue
        cmc = raw.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_rel = raw.sum()
        precision = raw.cumsum() / (np.arange(len(raw)) + 1)
        all_ap.append((precision * raw).sum() / num_rel)

    if not all_cmc:
        raise RuntimeError("No query identity appears in the gallery")
    cmc = np.asarray(all_cmc, dtype=np.float32).mean(axis=0)
    return cmc, float(np.mean(all_ap))


@torch.no_grad()
def evaluate(model, test_loader, num_query: int, device, flip: bool = True) -> dict[str, float]:
    feats, pids, cams = extract_features(model, test_loader, device, flip=flip)
    feats = F.normalize(feats, dim=1)
    qf, gf = feats[:num_query], feats[num_query:]
    distmat = (1 - qf @ gf.t()).numpy()  # cosine distance
    cmc, mAP = eval_market(
        distmat, pids[:num_query], pids[num_query:], cams[:num_query], cams[num_query:]
    )
    return {"mAP": mAP, "rank1": float(cmc[0]), "rank5": float(cmc[4]), "rank10": float(cmc[9])}
