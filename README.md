# KD-ReID

Knowledge distillation for person re-identification: transfer a large teacher (ResNet-50, CLIP-ReID ViT-B/16) into lightweight OSNet students while keeping retrieval accuracy.

Two student sizes are maintained:

| student | backbone params* | use case |
|---|---|---|
| **OSNet x0.25** | 0.2M | ultra-light / edge deployment |
| **OSNet x1.0** | 2.2M | still small, near-teacher accuracy |

\* inference model without the training-only identity classifier. ResNet-18 (11.2M) is kept as the classic homogeneous-distillation baseline.

## Main results

All numbers: last epoch (never the test-selected best checkpoint), cosine distance, flip TTA, **no re-ranking**. Full log with every run, config and commit: [`docs/experiments.md`](docs/experiments.md).

### Market-1501

| model | teacher / KD | mAP | R1 | seeds |
|---|---|---|---|---|
| ResNet-50 teacher | – | 86.5 ± 0.1 | 94.9 | 3 |
| CLIP-ReID ViT-B/16 (official weights) | – | 89.9 | 95.4 | – |
| OSNet x0.25 B\* (no KD) | – | 77.4 ± 0.4 | 90.8 | 3 |
| OSNet x0.25 D\* | R50 / logit KD | 82.6 ± 0.1 | 93.2 | 3 |
| **OSNet x0.25 F\*** | **R50 / DKD** | **85.0 ± 0.0** | **94.0** | 3 |
| OSNet x0.25 | CLIP-ReID / DKD | 83.45 ± 0.07 | 92.6 | 2 |
| OSNet x1.0 (no KD) | – | 85.7 ± 0.2 | 94.7 | 2 |
| OSNet x1.0 | R50 / DKD | 88.3 ± 0.0 | 95.2 | 2 |
| **OSNet x1.0** | **CLIP-ReID / DKD** | **89.5 ± 0.0** | **95.3** | 2 |
| ResNet-18 | R50 / DKD | 86.2 | 94.4 | 1 |

Negative results kept for the record (1 seed each): OSNet x0.25 ← OSNet x1.0 (6c) as teacher assistant 83.6; CLIP-ReID's own text-prototype classifier as the teacher (100% top-1 on the training set) gives OSNet x0.25 83.8 and OSNet x1.0 88.4, both below the R50 / linear-head results above. A teacher that fits the training labels better is not a better teacher.

- **DKD** (decoupled KD, α=1, β=8, T=4) beats classic logit KD on every student and dataset (OSNet x0.25: +2.4 mAP on Market, +5.1 on CUHK03-NP).
- **Teacher strength only helps a student large enough to absorb it.** The CLIP-ReID teacher is 3.4 mAP stronger than R50, yet it *lowers* the x0.25 student (−1.55) and *raises* the x1.0 student (+1.2). OSNet x1.0 ← CLIP-ReID reaches 89.5, within 0.4 of the CLIP-ReID teacher with ~1/40 of its parameters.

### CUHK03-NP (detected, 767/700 split)

| model | teacher / KD | mAP | R1 | seeds |
|---|---|---|---|---|
| ResNet-50 teacher | – | 61.4 | 62.4 | 1 |
| OSNet x0.25 B\* | – | 49.5 ± 0.3 | 52.3 | 3 |
| OSNet x0.25 D\* | R50 / logit KD | 55.6 ± 0.2 | 58.2 | 3 |
| **OSNet x0.25 F\*** | **R50 / DKD** | **61.0 ± 0.3** | **63.6** | 3 |
| OSNet x1.0 (no KD) / ← R50 DKD | – / R50 | 67.0 / 69.2 | 70.3 / 71.4 | 1 |
| ResNet-18 baseline / KD / DKD | – / R50 / R50 | 50.6 / 60.5 / 62.4 | 54.9 / 63.4 / 65.1 | 1 |

### Zero-shot cross-domain: Market-1501 → CUHK03-NP (mAP, 3 seeds)

CLIP-ReID teacher 39.2 · R50 teacher 5.5 ± 0.6 · OSNet x0.25 B\* 6.1 ± 0.1 · D\* 8.4 ± 0.5 · F\* 10.2 ± 0.9 · OSNet x1.0 no KD 6.6 ± 0.2 · x1.0 ← R50 8.8 ± 0.8 · **x1.0 ← CLIP-ReID 17.1 ± 0.6**. Distilled students generalise better than the R50 teacher, and part of the CLIP-ReID teacher's generalisation transfers to a student large enough to absorb it (a bigger student alone does not help). The x1.0 student reaches 99.6% of CLIP-ReID in-domain but only 44% cross-domain: cross-domain generalisation is the largest remaining gap.

### Label-free distillation on unlabeled target images (UDA, Market-1501 → CUHK03-NP)

OSNet x1.0 ← CLIP-ReID with an extra batch of unlabeled CUHK03-NP training images per iteration (cosine + in-batch similarity + logit KD on them), BatchNorm statistics re-estimated per domain: **Market 90.0 ± 0.0, CUHK03-NP 33.4 ± 0.6** (2 seeds), vs 89.5 / 17.1 without it and 22.8 ± 0.1 for AdaBN alone; 74% of the student's cross-domain gap to CLIP-ReID (39.2) is closed. The same extra batch drawn from Market images gives no cross-domain gain: the gain comes from distilling on target-domain images.

## Quick start

**Google Colab**: open `notebooks/colab_setup.ipynb` with a GPU runtime and run sections 1–5: they mount Drive, clone this repo, install it, unpack every dataset zip found in `MyDrive/KD-ReID/datasets/`, and run the smoke tests. Later sections hold one cell per experiment. Dataset splits are checked against the official sizes, so a wrong folder fails loudly instead of training on the wrong data.

**Kaggle** (background execution, no Drive): attach a dataset with the dataset zips and the CLIP-ReID / OSNet ImageNet checkpoints, add a `GH_TOKEN` secret, then use `notebooks/kaggle_run.ipynb` (train a list of `(config, seed)` runs in one session, then evaluate) or `notebooks/kaggle_eval.ipynb` (evaluate runs attached as inputs, including BN re-estimation). Every Drive path is overridden on the command line.

## Training

```bash
# teachers and baselines
python tools/train.py --config configs/teacher_r50.yaml
python tools/train.py --config configs/osnet/1e_lr1e-3_240ep.yaml      # OSNet x0.25 B* (no KD)

# OSNet x0.25 students (R50 teacher)
python tools/train.py --config configs/osnet/2d_kd_r50_logit.yaml      # D*: logit KD
python tools/train.py --config configs/osnet/3d_dkd_r50.yaml           # F*: DKD

# CLIP-ReID teacher (official checkpoint; run tools/check_teacher.py first)
python tools/train.py --config configs/clip/4a_dkd_clip.yaml           # -> OSNet x0.25
python tools/train.py --config configs/osnet_x1/6c_dkd_clip_x1.yaml    # -> OSNet x1.0
python tools/train.py --config configs/osnet_x1/6b_dkd_r50_x1.yaml     # x1.0 <- R50

# second dataset: same recipes on CUHK03-NP
python tools/train.py --config configs/cuhk03/osnet_fstar.yaml
python tools/train.py --config configs/cuhk03/osnet_x1_dkd_r50.yaml   # x1.0 <- R50 on CUHK03-NP

# label-free distillation on unlabeled images (method 4, configs/plan8)
python tools/train.py --config configs/plan8/8b_unl_cuhk03.yaml      # + unlabeled CUHK03-NP train images
python tools/train.py --config configs/plan8/8a_unl_market.yaml      # control: unlabeled Market images
```

Any config value can be overridden on the command line, e.g. `seed=1 output_dir=... kd.dkd.beta=16`. Training resumes automatically from `output_dir/last.pth`.

## Distillation losses (`cfg.kd.losses`)

| key | method |
|---|---|
| `logit` | Hinton KD on identity logits (KL, temperature T) |
| `dkd` | Decoupled KD (Zhao et al., CVPR 2022): target / non-target terms weighted separately |
| `feature` | L2 between projected student and teacher embeddings |
| `cosine` | 1 − cosine between projected student and teacher embeddings (O(1) for any width) |
| `similarity` | match in-batch cosine-similarity matrices |
| `simdist` | KL between in-batch similarity distributions |
| `rkd` | relational KD (distance + angle) |
| `attention` | attention transfer on feature maps |

Multiple teachers are supported (`teachers:` list with per-teacher weights / losses). Label-free terms (everything except `dkd`) can also be applied to an extra batch of unlabeled images with `unlabeled: {dataset | folder, batch_size, weight, losses}`; a dataset's training split is used with its labels discarded, and those terms are logged as `u_*`. A teacher whose identity head is nearly flat can be rescaled with `teacher.logit_scale` (temperature scaling); `tools/check_teacher.py` measures label alignment, fits the scale and checks the teacher's own retrieval accuracy. For CLIP-ReID the fitted value is 8.26.

## Evaluation and deployment

```bash
python tools/eval_matrix.py --runs <run dirs> --datasets market1501:/content/data cuhk03np:/content/data
python tools/eval_matrix.py --runs <run dirs> --bn-recal --datasets ...    # + per-domain BN statistics (AdaBN)
python tools/eval_matrix.py --teacher-configs configs/clip/teacher_eval.yaml --datasets ...  # official CLIP-ReID
python tools/export_package.py --run <run dir> --name <package>   # ONNX + weights + reid_infer.py
python tools/benchmark.py --models osnet_x0_25:512 osnet_x1_0:512 resnet50 clipreid_vit_b16 --onnx-dir onnx/
```

`--bn-recal` re-estimates every BatchNorm's running statistics on each test dataset's *training* images (no labels, training augmentation) before evaluating; re-estimating on the training dataset itself must reproduce the plain result (it does: 89.5 → 89.6 for OSNet x1.0). Exported packages are standalone (no dependency on this repo); the export checks the standalone model against the training model and ONNX against PyTorch.

Deployment cost (Colab T4 + 2 vCPU; CPU latency with ONNX Runtime, batch 1):

| model | params | GMACs | CPU ms | GPU fp16 ms / image (batch 32) | Market mAP |
|---|---|---|---|---|---|
| CLIP-ReID ViT-B/16 | 86.1M | 11.08 | 255.1 | 2.30 | 89.9 |
| ResNet-50 | 23.5M | 4.05 | 81.1 | 1.03 | 86.5 |
| OSNet x1.0 | 2.2M | 0.98 | 22.2 | 0.83 | 89.5 |
| OSNet x0.25 | 0.2M | 0.082 | 3.2 | 0.54 | 85.0 |

OSNet's advantage is on CPU / edge devices; on a GPU with PyTorch eager its many small kernels make it launch-bound.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q   # CPU, synthetic data
ruff check .
```
