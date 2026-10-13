"""Official Chameleon CIFAR-10 Type-1 trigger helpers.

The original implementation writes ``255`` directly into already normalized
tensor values.  This unusual behaviour is intentional here: it is not a raw
RGB white-pixel patch and must be shared by local training and ASR evaluation.
"""

from __future__ import annotations

import torch


TYPE1_COORDINATES: tuple[tuple[int, int], ...] = (
    (29, 29),
    (30, 28),
    (28, 30),
    (30, 30),
)
TYPE1_NORMALIZED_VALUE = 255.0


def apply_chameleon_type1_trigger(
    images: torch.Tensor,
    poison_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a copy with Chameleon's Type-1 value written post-normalization.

    ``poison_indices=None`` applies the trigger to every image, which is used
    by evaluation after the evaluator has excluded target-class examples.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [N, 3, H, W]")
    if images.shape[2] < 31 or images.shape[3] < 31:
        raise ValueError("Chameleon Type-1 requires image height and width of at least 31")
    if poison_indices is None:
        poison_indices = torch.arange(images.size(0), device=images.device, dtype=torch.long)
    if not isinstance(poison_indices, torch.Tensor) or poison_indices.dtype != torch.long:
        raise TypeError("poison_indices must be a torch.long tensor or None")
    if poison_indices.numel() and (
        (poison_indices < 0).any().item() or (poison_indices >= images.size(0)).any().item()
    ):
        raise ValueError("poison_indices must be valid batch indices")

    triggered = images.clone()
    for row, column in TYPE1_COORDINATES:
        triggered[poison_indices, :, row, column] = TYPE1_NORMALIZED_VALUE
    return triggered


def poison_chameleon_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_label: int,
    poison_per_batch: int,
    training: bool,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Poison the first configured positions and return copies plus indices.

    The first ``min(poison_per_batch, batch_size)`` samples are relabelled,
    exactly matching the official pixel-pattern injection order.  Training
    alone receives the official post-trigger Gaussian noise.
    """
    if labels.ndim != 1 or labels.size(0) != images.size(0):
        raise ValueError("labels must be one-dimensional and match images")
    if type(target_label) is not int or target_label < 0:
        raise ValueError("target_label must be a non-negative integer")
    if type(poison_per_batch) is not int or poison_per_batch <= 0:
        raise ValueError("poison_per_batch must be a positive integer")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    count = min(poison_per_batch, images.size(0))
    indices = torch.arange(count, device=images.device, dtype=torch.long)
    poisoned_images = apply_chameleon_type1_trigger(images, indices)
    poisoned_labels = labels.clone()
    if count:
        if training and noise_std:
            poisoned_images[indices] += torch.randn_like(poisoned_images[indices]) * noise_std
        poisoned_labels[indices] = target_label
    return poisoned_images, poisoned_labels, indices
