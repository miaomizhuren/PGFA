#!/usr/bin/env python3
"""CPU-only regression tests for compact/full prototype-bank indexing."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.adversarial_trigger import compute_prototype_loss


def _assert_finite_backward(loss: torch.Tensor | None, features: torch.Tensor) -> None:
    assert loss is not None
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_gtsrb_compact_bank_missing_class_36() -> None:
    class_ids = [class_id for class_id in range(43) if class_id != 36]
    assert class_ids.index(42) == 41
    prototypes = torch.randn(42, 8)
    valid_mask = torch.ones(43, dtype=torch.bool)
    valid_mask[36] = False
    features = torch.randn(3, 8, requires_grad=True)

    loss = compute_prototype_loss(
        features,
        prototypes,
        class_ids,
        target_proto_index=class_ids.index(0),
        temperature=0.01,
        mode="target_vs_source",
        source_labels=torch.tensor([42, 36, 37]),
        source_margin=0.7,
        valid_class_mask=valid_mask,
    )
    _assert_finite_backward(loss, features)


def test_cifar100_sparse_compact_bank() -> None:
    class_ids = [0, 7, 42, 99]
    prototypes = torch.randn(4, 8)
    valid_mask = torch.zeros(100, dtype=torch.bool)
    valid_mask[class_ids] = True
    features = torch.randn(3, 8, requires_grad=True)

    loss = compute_prototype_loss(
        features,
        prototypes,
        class_ids,
        target_proto_index=0,
        temperature=0.01,
        mode="target_vs_source",
        source_labels=torch.tensor([99, 50, 7]),
        source_margin=0.7,
        valid_class_mask=valid_mask,
    )
    _assert_finite_backward(loss, features)


def test_cifar10_complete_bank_semantics_unchanged() -> None:
    class_ids = list(range(10))
    prototypes = torch.randn(10, 8)
    features = torch.randn(5, 8)
    source_labels = torch.tensor([1, 2, 3, 4, 5])
    kwargs = dict(
        features=features,
        prototypes=prototypes,
        class_ids=class_ids,
        target_proto_index=0,
        temperature=0.01,
        mode="target_vs_source",
        source_labels=source_labels,
        source_margin=0.7,
    )
    unmasked = compute_prototype_loss(**kwargs)
    masked = compute_prototype_loss(
        **kwargs, valid_class_mask=torch.ones(10, dtype=torch.bool)
    )
    assert unmasked is not None and masked is not None
    assert torch.allclose(unmasked, masked)


def test_full_bank_mask_and_missing_classes() -> None:
    class_ids = list(range(43))
    prototypes = torch.randn(43, 8)
    valid_mask = torch.ones(43, dtype=torch.bool)
    valid_mask[36] = False
    features = torch.randn(2, 8, requires_grad=True)
    loss = compute_prototype_loss(
        features,
        prototypes,
        class_ids,
        target_proto_index=0,
        temperature=0.01,
        mode="target_vs_source",
        source_labels=torch.tensor([42, 36]),
        source_margin=0.7,
        valid_class_mask=valid_mask,
    )
    _assert_finite_backward(loss, features)


def test_safe_missing_target_and_source_fallbacks() -> None:
    prototypes = torch.randn(2, 8)
    features = torch.randn(2, 8)
    valid_mask = torch.zeros(43, dtype=torch.bool)
    valid_mask[[1, 42]] = True

    assert compute_prototype_loss(
        features,
        prototypes,
        [1, 42],
        target_proto_index=None,
        temperature=0.01,
        mode="target_vs_all",
        valid_class_mask=valid_mask,
    ) is None
    assert compute_prototype_loss(
        features,
        prototypes,
        [0, 42],
        target_proto_index=0,
        temperature=0.01,
        mode="target_vs_source",
        source_labels=torch.tensor([36, 36]),
        valid_class_mask=torch.tensor([True] + [False] * 41 + [True]),
    ) is None


def main() -> None:
    torch.manual_seed(2026)
    test_gtsrb_compact_bank_missing_class_36()
    test_cifar100_sparse_compact_bank()
    test_cifar10_complete_bank_semantics_unchanged()
    test_full_bank_mask_and_missing_classes()
    test_safe_missing_target_and_source_fallbacks()
    print("PASS: compact/full prototype indexing and missing-class fallbacks")


if __name__ == "__main__":
    main()
