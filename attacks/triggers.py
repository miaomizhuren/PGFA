"""Normalized CIFAR-10 patch-trigger helpers for backdoor experiments."""

import torch

from attacks.adversarial_trigger import generate_adversarial_trigger
from data.dataset import CIFAR10_MEAN, CIFAR10_STD


def clamp_normalized_cifar10_rgb(images: torch.Tensor) -> torch.Tensor:
    """Clamp normalized CIFAR-10 tensors to the raw RGB [0, 1] domain.

    The project applies UAPs after ``Normalize``. This helper is opt-in so
    historical adversarial-trigger runs retain their exact behaviour.
    """
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("normalized CIFAR-10 RGB clipping requires [N, 3, H, W] images")
    mean = torch.tensor(CIFAR10_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return torch.maximum(torch.minimum(images, (1.0 - mean) / std), -mean / std)


def normalize_pixel_value(
    pixel_value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert an RGB pixel value from [0, 1] to CIFAR-10 normalized values."""
    if isinstance(pixel_value, bool) or not isinstance(pixel_value, (int, float)):
        raise TypeError("pixel_value must be a number")
    if not 0.0 <= pixel_value <= 1.0:
        raise ValueError("pixel_value must be between 0.0 and 1.0")

    normalized = [
        (pixel_value - mean) / std
        for mean, std in zip(CIFAR10_MEAN, CIFAR10_STD)
    ]
    return torch.tensor(normalized, device=device, dtype=dtype).view(3, 1, 1)


def apply_patch_trigger(
    images: torch.Tensor,
    poison_indices: torch.Tensor,
    patch_size: int = 3,
    pixel_value: float = 1.0,
) -> torch.Tensor:
    """Return copied images with a normalized patch in selected bottom-right corners."""
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim != 4:
        raise ValueError("images must have shape [N, 3, H, W]")
    if images.shape[1] != 3:
        raise ValueError("images must have three channels")
    if not isinstance(poison_indices, torch.Tensor):
        raise TypeError("poison_indices must be a torch.Tensor")
    if poison_indices.dtype != torch.long:
        raise TypeError("poison_indices must have dtype torch.long")
    if type(patch_size) is not int:
        raise TypeError("patch_size must be an integer")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if isinstance(pixel_value, bool) or not isinstance(pixel_value, (int, float)):
        raise TypeError("pixel_value must be a number")
    if not 0.0 <= pixel_value <= 1.0:
        raise ValueError("pixel_value must be between 0.0 and 1.0")

    batch_size, _, height, width = images.shape
    if patch_size > height or patch_size > width:
        raise ValueError("patch_size cannot exceed image height or width")
    if poison_indices.numel() and (
        (poison_indices < 0).any().item()
        or (poison_indices >= batch_size).any().item()
    ):
        raise ValueError("poison_indices must be valid batch indices")

    poisoned_images = images.clone()
    if poison_indices.numel() == 0:
        return poisoned_images

    normalized_value = normalize_pixel_value(
        pixel_value,
        images.device,
        images.dtype,
    )
    poisoned_images[
        poison_indices,
        :,
        height - patch_size : height,
        width - patch_size : width,
    ] = normalized_value
    return poisoned_images


def apply_trigger(
    images: torch.Tensor,
    trigger_type: str = "patch",
    *,
    poison_indices: torch.Tensor | None = None,
    patch_size: int = 3,
    pixel_value: float = 1.0,
    model: torch.nn.Module | None = None,
    target_label: int | None = None,
    epsilon: float | None = None,
    steps: int | None = None,
    alpha: float | None = None,
    perturbation: torch.Tensor | None = None,
    clamp_to_valid_rgb: bool = False,
) -> torch.Tensor:
    """Apply a patch trigger or a shared adversarial trigger to ``images``.

    Patch arguments intentionally match :func:`apply_patch_trigger`.  For an
    adversarial trigger, either supply ``perturbation`` or supply ``model``,
    ``target_label``, ``epsilon``, ``steps``, and ``alpha`` to generate it.
    """
    if trigger_type == "patch":
        if poison_indices is None:
            raise ValueError("poison_indices is required for a patch trigger")
        return apply_patch_trigger(images, poison_indices, patch_size, pixel_value)

    if trigger_type != "adversarial":
        raise ValueError("trigger_type must be 'patch' or 'adversarial'")
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if images.ndim != 4:
        raise ValueError("images must have shape [N, C, H, W]")

    if perturbation is None:
        missing = [
            name
            for name, value in {
                "model": model,
                "target_label": target_label,
                "epsilon": epsilon,
                "steps": steps,
                "alpha": alpha,
            }.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "adversarial trigger generation requires " + ", ".join(missing)
            )
        perturbation = generate_adversarial_trigger(
            model=model,
            images=images,
            target_label=target_label,
            epsilon=epsilon,
            steps=steps,
            alpha=alpha,
        )

    if not isinstance(perturbation, torch.Tensor):
        raise TypeError("perturbation must be a torch.Tensor")
    if perturbation.device != images.device:
        raise ValueError("perturbation must be on the same device as images")
    if perturbation.dtype != images.dtype:
        raise TypeError("perturbation must have the same dtype as images")
    if perturbation.ndim == 3:
        if tuple(perturbation.shape) != tuple(images.shape[1:]):
            raise ValueError("3-D perturbation must have shape [C, H, W]")
        perturbation = perturbation.unsqueeze(0)
    if perturbation.ndim != 4 or perturbation.shape[1:] != images.shape[1:]:
        raise ValueError("perturbation must have shape [C,H,W] or [1 or N,C,H,W]")
    if perturbation.shape[0] not in (1, images.shape[0]):
        raise ValueError("perturbation batch dimension must be 1 or match images")
    triggered = images + perturbation
    return clamp_normalized_cifar10_rgb(triggered) if clamp_to_valid_rgb else triggered
