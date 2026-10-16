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
| OSNet x1.0 | R50 / DKD | 88.3 | 95.3 | 1 |
| **OSNet x1.0** | **CLIP-ReID / DKD** | **89.5** | **95.4** | 1 |
| ResNet-18 | R50 / DKD | 86.2 | 94.4 | 1 |

- **DKD** (decoupled KD, α=1, β=8, T=4) beats classic logit KD on every student and dataset (OSNet x0.25: +2.4 mAP on Market, +5.1 on CUHK03-NP).
- **Teacher strength only helps a student large enough to absorb it.** The CLIP-ReID teacher is 3.4 mAP stronger than R50, yet it *lowers* the x0.25 student (−1.55) and *raises* the x1.0 student (+1.2). OSNet x1.0 ← CLIP-ReID reaches 89.5, within 0.4 of the CLIP-ReID teacher with ~1/40 of its parameters.

### CUHK03-NP (detected, 767/700 split)

| model | teacher / KD | mAP | R1 |
|---|---|---|---|
| ResNet-50 teacher | – | 61.4 | 62.4 |
| OSNet x0.25 B\* | – | 49.3 | 52.4 |
| OSNet x0.25 D\* | R50 / logit KD | 55.8 | 58.4 |
| **OSNet x0.25 F\*** | **R50 / DKD** | **60.9** | **63.9** |
| ResNet-18 baseline / KD / DKD | – / R50 / R50 | 50.6 / 60.5 / 62.4 | 54.9 / 63.4 / 65.1 |

### Zero-shot cross-domain: Market-1501 → CUHK03-NP (mAP, 3 seeds)

R50 teacher 5.5 ± 0.6 · OSNet x0.25 B\* 6.1 ± 0.1 · D\* 8.4 ± 0.5 · **F\* 10.2 ± 0.9**. The distilled 0.2M student generalises better than its own teacher.

## Quick start (Google Colab)

Open `notebooks/colab_setup.ipynb` in Colab with a GPU runtime and run it top to bottom: it mounts Drive, clones this repo, installs it, unpacks every dataset zip found in `MyDrive/KD-ReID/datasets/`, and runs the smoke tests. Dataset splits are checked against the official sizes, so a wrong folder fails loudly instead of training on the wrong data.

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
```

Any config value can be overridden on the command line, e.g. `seed=1 output_dir=... kd.dkd.beta=16`. Training resumes automatically from `output_dir/last.pth`.

## Distillation losses (`cfg.kd.losses`)

| key | method |
|---|---|
| `logit` | Hinton KD on identity logits (KL, temperature T) |
| `dkd` | Decoupled KD (Zhao et al., CVPR 2022): target / non-target terms weighted separately |
| `feature` | L2 between projected student and teacher embeddings |
| `similarity` | match in-batch cosine-similarity matrices |
| `simdist` | KL between in-batch similarity distributions |
| `rkd` | relational KD (distance + angle) |
| `attention` | attention transfer on feature maps |

Multiple teachers are supported (`teachers:` list with per-teacher weights / losses). A teacher whose identity head is nearly flat can be rescaled with `teacher.logit_scale` (temperature scaling); `tools/check_teacher.py` measures label alignment, fits the scale and checks the teacher's own retrieval accuracy. For CLIP-ReID the fitted value is 8.26.

## Evaluation and deployment

```bash
python tools/eval_matrix.py --runs <run dirs> --datasets market1501:/content/data cuhk03np:/content/data
python tools/export_package.py --run <run dir> --name <package>   # ONNX + weights + reid_infer.py
python tools/benchmark.py --models osnet_x0_25:512 osnet_x1_0:512 resnet50 --onnx-dir onnx/
```

Exported packages are standalone (no dependency on this repo); the export checks the standalone model against the training model and ONNX against PyTorch.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q   # CPU, synthetic data
ruff check .
```
