#!/usr/bin/env python3
"""Render a horizontal, grayscale-delta PGFA fixed-UAP visualization (read-only)."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_paper_figures"
DELTA_PATH = ROOT / "outputs" / "pbu_force_clean_stage_sensitivity" / "clean3000_attack3001_3200_observe3201_3700" / "checkpoints" / "delta_round_3200.pt"
TARGET_LABEL = 0
SELECTION_SEED = 2026
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=images.dtype).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def select_samples(labels: list[int]) -> list[int]:
    """Deterministically choose one each from classes 1–9 plus one extra non-target sample."""
    rng = random.Random(SELECTION_SEED)
    chosen: list[int] = []
    for label in range(1, 10):
        candidates = [i for i, value in enumerate(labels) if value == label]
        chosen.append(rng.choice(candidates))
    extra_label = rng.choice(list(range(1, 10)))
    candidates = [i for i, value in enumerate(labels) if value == extra_label and i not in chosen]
    chosen.append(rng.choice(candidates))
    return chosen


def grayscale_zero_centered_delta(delta: torch.Tensor) -> torch.Tensor:
    """Convert normalized-coordinate delta to display-RGB, then zero-center in grayscale."""
    std = torch.tensor(CIFAR10_STD, dtype=delta.dtype).view(1, 3, 1, 1)
    delta_display_rgb = delta * std  # Same image-display coordinate change as denormalization.
    luminance = (0.2126 * delta_display_rgb[0, 0] + 0.7152 * delta_display_rgb[0, 1] + 0.0722 * delta_display_rgb[0, 2])
    scale = luminance.abs().amax().clamp_min(1e-12)
    # Deliberately restrict contrast to [0.25, 0.75] to avoid overstating visibility.
    return (0.5 + 0.25 * luminance / scale).clamp(0.0, 1.0)


def main() -> None:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_set = datasets.CIFAR10(root=ROOT / "datasets", train=False, transform=transform, download=False)
    indices = select_samples([int(label) for label in test_set.targets])
    images, labels = zip(*(test_set[index] for index in indices))
    labels = [int(label) for label in labels]
    if any(label == TARGET_LABEL for label in labels):
        raise ValueError("Target-class samples must be excluded")
    clean_normalized = torch.stack(images)

    delta = torch.load(DELTA_PATH, map_location="cpu", weights_only=False)
    if not isinstance(delta, torch.Tensor) or tuple(delta.shape) != (1, 3, 32, 32):
        raise ValueError(f"Expected [1, 3, 32, 32] tensor; got {type(delta)} {getattr(delta, 'shape', None)}")
    delta = delta.to(dtype=clean_normalized.dtype)
    # Raw delta is used here exactly as in the attack code; display_delta is never used for triggering.
    triggered_normalized = clean_normalized + delta
    clean_rgb, triggered_rgb = denormalize(clean_normalized), denormalize(triggered_normalized)
    delta_gray = grayscale_zero_centered_delta(delta)

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(3, 10, figsize=(15.0, 4.85), facecolor="white")
    row_labels = ("Clean", "Shared perturbation δ", "Triggered image x+δ")
    for col in range(10):
        panels = (clean_rgb[col], delta_gray, triggered_rgb[col])
        for row, panel in enumerate(panels):
            ax = axes[row, col]
            if row == 1:
                ax.imshow(panel.numpy(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            else:
                ax.imshow(panel.permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=10.5, rotation=90, labelpad=13, va="center")
    fig.subplots_adjust(left=0.082, right=0.995, top=0.99, bottom=0.025, wspace=0.035, hspace=0.075)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "fixed_uap_visualization_pgfa_cifar10_horizontal_graydelta.png"
    pdf = OUT / "fixed_uap_visualization_pgfa_cifar10_horizontal_graydelta.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    sample_text = ", ".join(f"{idx} ({CLASS_NAMES[label]}, label {label})" for idx, label in zip(indices, labels))
    summary = OUT / "fixed_uap_visualization_pgfa_cifar10_horizontal_graydelta_summary.md"
    summary.write_text("\n".join([
        "# Fixed UAP qualitative visualization: horizontal grayscale-delta revision", "",
        "## Inputs", "", f"- Delta: `{DELTA_PATH.relative_to(ROOT)}`", f"- Delta tensor shape: `{tuple(delta.shape)}`", "- Test set: CIFAR-10 test split (`datasets/cifar-10-batches-py`)", "",
        "## Samples and layout", "", f"- Deterministic selection seed: `{SELECTION_SEED}`", f"- Test sample indices: `{indices}`", f"- Labels: `{labels}`", f"- Labels with names: {sample_text}", f"- Target class 0 excluded: **yes**", f"- Number of samples: `{len(indices)}`", "- Layout: **3 × 10** horizontal single figure (not split into blocks).", "",
        "## Perturbation and triggering", "", "- Perturbation visualization: **grayscale**, **zero-centered**, and rescaled for visualization only.",
        "- The saved normalized-space delta is first converted to image-display space by multiplying each channel by the CIFAR-10 std. Its luminance is linearly mapped with zero as neutral gray (0.5); contrast is deliberately restricted to `[0.25, 0.75]` so negative/positive values are darker/lighter without overstating visual visibility.",
        "- Triggered images use the **raw saved delta**, not the displayed grayscale/rescaled delta: normalized clean image + raw delta, followed by CIFAR-10 denormalization and display clipping to `[0, 1]`.", "",
        "## Outputs", "", f"- PNG: `{png.relative_to(ROOT)}`", f"- PDF: `{pdf.relative_to(ROOT)}`", "",
        "## Suggested caption", "", "Qualitative visualization of the fixed targeted UAP on CIFAR-10. From top to bottom, the rows show clean images, the shared perturbation δ rescaled for visualization, and the corresponding triggered images x+δ. The same fixed perturbation is applied to different non-target samples. The perturbation visualization is zero-centered and rescaled only for display, while the triggered images are generated using the original raw δ.", "",
        "No training was started and no original checkpoint, metric, CSV, or SQLite result was modified.",
    ]) + "\n")


if __name__ == "__main__":
    main()
