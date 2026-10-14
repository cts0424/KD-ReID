"""label_stats must tell a label-order mismatch apart from a weak classifier head."""

import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "check_teacher", Path(__file__).resolve().parents[1] / "tools/check_teacher.py"
)
check_teacher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_teacher)


def _onehot_logits(pred, n):
    return torch.nn.functional.one_hot(pred, n).float() * 10


def test_aligned_teacher():
    labels = torch.arange(10).repeat(5)
    s = check_teacher.label_stats(_onehot_logits(labels, 10), labels)
    assert s["top1"] == 1.0 and s["remapped_top1"] == 1.0 and s["max_prob"] > 0.99


def test_permuted_labels_are_recovered_by_remapping():
    labels = torch.arange(10).repeat(5)
    perm = torch.roll(torch.arange(10), 1)
    s = check_teacher.label_stats(_onehot_logits(perm[labels], 10), labels)
    assert s["top1"] == 0.0 and s["remapped_top1"] == 1.0


def test_scattered_errors_are_not_recovered():
    torch.manual_seed(0)
    labels = torch.arange(20).repeat(20)
    pred = torch.where(torch.rand(len(labels)) < 0.6, labels, torch.randint(0, 20, labels.shape))
    s = check_teacher.label_stats(_onehot_logits(pred, 20), labels)
    assert 0.55 < s["top1"] < 0.75 and s["remapped_top1"] - s["top1"] < 0.05
