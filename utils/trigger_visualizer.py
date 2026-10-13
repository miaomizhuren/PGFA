"""Utilities for saving CIFAR-10 trigger comparison images."""

from pathlib import Path

import torch
import torch.nn.functional as functional
from torchvision.utils import make_grid, save_image

from data.dataset import CIFAR10_MEAN, CIFAR10_STD


UPSCALE_FACTOR = 4


def _denormalize(images: torch.Tensor, mean_values=CIFAR10_MEAN, std_values=CIFAR10_STD) -> torch.Tensor:
    """Convert normalized RGB images to clamped values in [0, 1]."""
    mean_values = CIFAR10_MEAN if mean_values is None else mean_values
    std_values = CIFAR10_STD if std_values is None else std_values
    mean = torch.tensor(mean_values, device=images.device, dtype=images.dtype)
    std = torch.tensor(std_values, device=images.device, dtype=images.dtype)
    return (images * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)).clamp(0.0, 1.0)


def save_trigger_comparison(
    original_images: torch.Tensor,
    triggered_images: torch.Tensor,
    delta: torch.Tensor,
    round_index: int,
    output_dir: str | Path = "outputs",
    num_samples: int = 5,
    mean_values=CIFAR10_MEAN,
    std_values=CIFAR10_STD,
) -> Path:
    """Save denormalized originals, triggered images, and amplified deltas.

    Files are written to ``<output_dir>/trigger_visualization/round_XX``.
    """
    if not isinstance(original_images, torch.Tensor):
        raise TypeError("original_images must be a torch.Tensor")
    if not isinstance(triggered_images, torch.Tensor):
        raise TypeError("triggered_images must be a torch.Tensor")
    if not isinstance(delta, torch.Tensor):
        raise TypeError("delta must be a torch.Tensor")
    if original_images.ndim != 4 or triggered_images.ndim != 4:
        raise ValueError("image tensors must have shape [N, C, H, W]")
    if original_images.shape != triggered_images.shape:
        raise ValueError("original_images and triggered_images must have the same shape")
    if original_images.shape[1] != 3:
        raise ValueError("CIFAR-10 images must have three channels")
    if delta.ndim != 4 or delta.shape[1:] != original_images.shape[1:]:
        raise ValueError("delta must have shape [1 or N, C, H, W]")
    if delta.shape[0] not in (1, original_images.shape[0]):
        raise ValueError("delta batch dimension must be 1 or match the image batch")
    if type(round_index) is not int or round_index <= 0:
        raise ValueError("round_index must be a positive integer")
    if type(num_samples) is not int or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    count = min(num_samples, original_images.shape[0])
    destination = Path(output_dir) / "trigger_visualization" / f"round_{round_index:02d}"
    destination.mkdir(parents=True, exist_ok=True)

    originals = _denormalize(original_images[:count].detach(), mean_values, std_values).cpu()
    triggered = _denormalize(triggered_images[:count].detach(), mean_values, std_values).cpu()
    visual_delta = delta.detach().abs()
    if visual_delta.shape[0] == 1:
        visual_delta = visual_delta.expand(count, -1, -1, -1)
    else:
        visual_delta = visual_delta[:count]
    visual_delta = visual_delta.cpu()
    scale = visual_delta.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    visual_delta = (visual_delta / scale).clamp(0.0, 1.0)

    for index in range(count):
        save_image(originals[index], destination / f"original_{index}.png")
        save_image(triggered[index], destination / f"triggered_{index}.png")
        save_image(visual_delta[index], destination / f"delta_{index}.png")

    comparison_rows = torch.stack(
        (originals, triggered, visual_delta),
        dim=1,
    ).reshape(count * 3, *originals.shape[1:])
    comparison_grid = make_grid(comparison_rows, nrow=3, padding=4)
    save_image(comparison_grid, destination / "comparison_grid.png")

    upscaled_rows = functional.interpolate(
        comparison_rows,
        scale_factor=UPSCALE_FACTOR,
        mode="nearest",
    )
    upscaled_comparison_grid = make_grid(upscaled_rows, nrow=3, padding=4)
    save_image(
        upscaled_comparison_grid,
        destination / "comparison_grid_upscaled.png",
    )

    return destination
