#!/usr/bin/env python3
"""Create a read-only CIFAR-10 qualitative visualization for the saved PGFA UAP."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DELTA_PATH = ROOT / "outputs" / "pbu_force_clean_stage_sensitivity" / "clean3000_attack3001_3200_observe3201_3700" / "checkpoints" / "delta_round_3200.pt"
OUT = ROOT / "outputs" / "final_paper_figures"
TARGET_LABEL = 0
CLASS_NAMES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# Matches data.dataset.CIFAR10_MEAN / CIFAR10_STD used in the main experiment.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=images.dtype).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def select_diverse_non_target_indices(labels: list[int]) -> list[int]:
    """Choose 10 deterministic test examples: classes 1–9, then a second class-1 sample."""
    selected: list[int] = []
    for wanted in range(1, 10):
        selected.append(next(index for index, label in enumerate(labels) if label == wanted))
    selected.append(next(index for index, label in enumerate(labels) if label == 1 and index != selected[0]))
    return selected


def main() -> None:
    # CIFAR-10's test transform matches the main experiment: ToTensor then normalization.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_set = datasets.CIFAR10(root=ROOT / "datasets", train=False, transform=transform, download=False)
    indices = select_diverse_non_target_indices([int(label) for label in test_set.targets])
    images, labels = zip(*(test_set[index] for index in indices))
    clean_normalized = torch.stack(images)
    labels = [int(label) for label in labels]

    delta = torch.load(DELTA_PATH, map_location="cpu", weights_only=False)
    if not isinstance(delta, torch.Tensor) or tuple(delta.shape) != (1, 3, 32, 32):
        raise ValueError(f"Expected delta tensor [1, 3, 32, 32], got {type(delta)} {getattr(delta, 'shape', None)}")
    delta = delta.to(dtype=clean_normalized.dtype)
    triggered_normalized = clean_normalized + delta
    clean_rgb, triggered_rgb = denormalize(clean_normalized), denormalize(triggered_normalized)

    # Signed normalized-coordinate delta, remapped from [-max_abs, +max_abs] to [0, 1] for display only.
    max_abs = delta.abs().amax().clamp_min(1e-12)
    delta_visual = ((delta / max_abs) + 1.0).div(2.0).clamp(0.0, 1.0)[0]

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    # Two 3 x 5 blocks: each block uses the same shared perturbation for all five samples.
    fig, axes = plt.subplots(6, 5, figsize=(9.2, 9.7), facecolor="white")
    row_labels = ("Clean", "Shared perturbation δ", "Triggered image x+δ")
    for block in range(2):
        for col in range(5):
            sample = block * 5 + col
            display_rows = (clean_rgb[sample], delta_visual, triggered_rgb[sample])
            for local_row, image in enumerate(display_rows):
                ax = axes[block * 3 + local_row, col]
                ax.imshow(image.permute(1, 2, 0).numpy(), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("white")
                for spine in ax.spines.values(): spine.set_visible(False)
                if col == 0:
                    ax.set_ylabel(row_labels[local_row], fontsize=10, rotation=90, labelpad=12, va="center")
    fig.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.02, wspace=0.045, hspace=0.11)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "fixed_uap_visualization_pgfa_cifar10.png"
    pdf = OUT / "fixed_uap_visualization_pgfa_cifar10.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    samples = ", ".join(f"{idx} ({CLASS_NAMES[label]}, label {label})" for idx, label in zip(indices, labels))
    summary = OUT / "fixed_uap_visualization_pgfa_cifar10_summary.md"
    summary.write_text("\n".join([
        "# Fixed UAP visualization: PGFA on CIFAR-10", "",
        "## Read-only inputs", "",
        f"- Delta: `{DELTA_PATH.relative_to(ROOT)}`", "- Test set: `datasets/cifar-10-batches-py` (CIFAR-10 test split)", "",
        "## Samples", "", f"- Count: {len(indices)}", f"- Target class: {TARGET_LABEL} ({CLASS_NAMES[TARGET_LABEL]}); excluded: **yes**.", f"- Test sample indices and labels: {samples}", "",
        "## Visualization", "", "- The same fixed delta is applied to every selected non-target test sample in normalized CIFAR-10 coordinates.",
        "- Clean and triggered images are denormalized with CIFAR-10 mean `(0.4914, 0.4822, 0.4465)` and std `(0.2470, 0.2435, 0.2616)`, then clamped to `[0, 1]` for display.",
        "- The perturbation row is **rescaled for visualization only** using signed max-absolute normalization: `display_delta = 0.5 * (delta / max(abs(delta)) + 1)`. It is not an image-space trigger and does not alter the saved delta.", "",
        "## Outputs", "", f"- PNG: `{png.relative_to(ROOT)}`", f"- PDF: `{pdf.relative_to(ROOT)}`", "",
        "No training was started and no checkpoint, metric, CSV, or existing experimental result was modified.",
    ]) + "\n")


if __name__ == "__main__":
    main()
