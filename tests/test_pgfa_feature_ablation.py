import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.adversarial_trigger import compute_feature_regularization


def _assert_gradient(loss, features):
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0


def test_margin_target_only_and_alignment_are_differentiable():
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    for variant in ("margin", "target_only", "feature_alignment"):
        features = torch.tensor([[0.2, 0.8], [0.6, 0.4]], requires_grad=True)
        reference = torch.tensor([1.0, 0.0], requires_grad=True)
        loss = compute_feature_regularization(
            features, variant, prototypes, [2, 7], 0, 0.1, "target_vs_source",
            torch.tensor([7, 7]), 0.2, torch.tensor([False] * 8).scatter_(0, torch.tensor([2, 7]), True), reference,
        )
        assert loss is not None and torch.isfinite(loss)
        _assert_gradient(loss, features)
        assert reference.grad is None


def test_target_only_is_one_minus_cosine_and_missing_target_is_explicit():
    features = torch.tensor([[1.0, 0.0]], requires_grad=True)
    prototypes = torch.tensor([[0.0, 1.0]])
    loss = compute_feature_regularization(features, "target_only", prototypes, [5], 0)
    assert torch.allclose(loss, torch.tensor(1.0))
    assert compute_feature_regularization(features, "target_only", prototypes, [5], None) is None


def test_feature_alignment_does_not_use_prototype_mean():
    features = torch.tensor([[0.8, 0.2]], requires_grad=True)
    reference = torch.tensor([1.0, 0.0], requires_grad=True)
    loss = compute_feature_regularization(features, "feature_alignment", target_reference_feature=reference)
    assert loss.item() > 0.0
    _assert_gradient(loss, features)


def test_feature_alignment_migrates_cached_cpu_reference_and_preserves_detach():
    features = torch.tensor([[0.8, 0.2]], dtype=torch.float64, requires_grad=True)
    cached_cpu_reference = torch.tensor([1.0, 0.0], dtype=torch.float32, requires_grad=True)
    loss = compute_feature_regularization(
        features, "feature_alignment", target_reference_feature=cached_cpu_reference
    )
    assert loss is not None
    assert loss.dtype == torch.float64
    _assert_gradient(loss, features)
    assert cached_cpu_reference.grad is None


if __name__ == "__main__":
    test_margin_target_only_and_alignment_are_differentiable()
    test_target_only_is_one_minus_cosine_and_missing_target_is_explicit()
    test_feature_alignment_does_not_use_prototype_mean()
    test_feature_alignment_migrates_cached_cpu_reference_and_preserves_detach()
    print("PASS: PGFA feature ablation CPU tests")
