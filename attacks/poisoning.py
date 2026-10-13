"""Batch-level label poisoning helpers for federated backdoor experiments."""

import torch


def select_poison_indices(
    batch_size: int,
    poison_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Randomly select unique batch indices to poison."""
    if type(batch_size) is not int:
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(poison_ratio, bool) or not isinstance(poison_ratio, (int, float)):
        raise TypeError("poison_ratio must be a number")
    if not 0.0 <= poison_ratio <= 1.0:
        raise ValueError("poison_ratio must be between 0.0 and 1.0")

    if poison_ratio == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    poison_count = min(batch_size, max(1, round(batch_size * poison_ratio)))
    return torch.randperm(batch_size, device=device)[:poison_count]


def relabel_poisoned_samples(
    labels: torch.Tensor,
    poison_indices: torch.Tensor,
    target_label: int,
) -> torch.Tensor:
    """Return copied labels with the selected samples set to the target label."""
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if not isinstance(poison_indices, torch.Tensor):
        raise TypeError("poison_indices must be a torch.Tensor")
    if poison_indices.dtype != torch.long:
        raise TypeError("poison_indices must have dtype torch.long")
    if type(target_label) is not int:
        raise TypeError("target_label must be an integer")
    if target_label < 0:
        raise ValueError("target_label must be non-negative")

    poisoned_labels = labels.clone()
    poisoned_labels[poison_indices] = target_label
    return poisoned_labels


def choose_non_target_indices(
    labels: torch.Tensor,
    target_label: int,
) -> torch.Tensor:
    """Return indices whose labels differ from the target label."""
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")
    if type(target_label) is not int:
        raise TypeError("target_label must be an integer")
    if target_label < 0:
        raise ValueError("target_label must be non-negative")

    return torch.nonzero(labels != target_label, as_tuple=False).flatten()
