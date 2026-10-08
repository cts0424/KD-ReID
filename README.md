# KD-ReID

Knowledge distillation for person re-identification: compress a heavy teacher (ResNet-50) into a lightweight student (ResNet-18 / MobileNetV3) while keeping retrieval accuracy.

## Quick start (Google Colab)

Open `notebooks/colab_setup.ipynb` in Colab with a GPU runtime and run it top to bottom. It mounts Drive, clones this repo, installs it, unpacks Market-1501 and runs the smoke tests.

## Pipeline

```bash
python tools/train.py --config configs/teacher_r50.yaml            # 1. teacher (upper bound)
python tools/train.py --config configs/student_r18_baseline.yaml   # 2. student alone (lower bound)
python tools/train.py --config configs/student_r18_kd.yaml         # 3. student + KD
```

Any config value can be overridden on the command line, e.g. `kd.losses.feature=1.0 kd.temperature=2`.

## Distillation losses

| key | method |
|---|---|
| `logit` | Hinton KD on identity logits (KL, temperature T) |
| `feature` | L2 between projected student and teacher embeddings |
| `similarity` | match in-batch cosine-similarity matrices (relational KD) |

## Tests

```bash
pip install -e ".[dev]"
python -m pytest -q   # CPU, synthetic data, ~10 s
```

Experiment log: [`docs/experiments.md`](docs/experiments.md).
