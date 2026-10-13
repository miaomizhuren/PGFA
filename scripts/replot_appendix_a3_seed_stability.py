#!/usr/bin/env python3
"""Render only Appendix Figure A3 (revised2) from existing seed-summary CSVs."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_paper_figures"
RAW_CSV = ROOT / "outputs" / "final_paper_summary_20260918" / "02_seed_stability_cifar10.csv"
STATS_CSV = ROOT / "outputs" / "final_paper_summary_20260918" / "02_seed_stability_cifar10_mean_std.csv"
BLUE = "#2F6F8F"
DARK_BLUE = "#1D4E63"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#D5D9DC", linewidth=0.45, alpha=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=8.5, width=0.75, length=3)


def main() -> None:
    raw, stats = read_csv(RAW_CSV), read_csv(STATS_CSV)
    specs = [
        ("Post_AUC_1_500", "(a) PA-AUC", "PA-AUC"),
        ("ASR_post_500", "(b) ASR@+500", "ASR@+500 (%)"),
        ("ASR_ge_80_duration", "(c) N80", "N80 (rounds)"),
    ]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.35))
    x = np.arange(1, 6)
    jitter = np.linspace(-0.035, 0.035, len(x))

    for ax, (metric, title, ylabel) in zip(axes, specs):
        stat = next(row for row in stats if row["Metric"] == metric)
        mean, std = value(stat, "Mean"), value(stat, "Std")
        y = np.array([value(row, metric) for row in raw])
        # The band and line span the panel, not an individual seed position.
        ax.axhspan(mean - std, mean + std, xmin=0, xmax=1, color=BLUE, alpha=0.16, zorder=1)
        ax.axhline(mean, color=DARK_BLUE, linewidth=1.8, zorder=2)
        ax.scatter(x + jitter, y, s=32, color=BLUE, edgecolors="white", linewidths=0.55, alpha=0.9, zorder=3)
        style(ax)
        ax.set_xlim(0.7, 5.3)
        ax.set_xticks(x)
        ax.set_xlabel("Seed index")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10.5, pad=7)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "appendix_figure_a3_seed_stability_revised2.png"
    pdf = OUT / "appendix_figure_a3_seed_stability_revised2.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = OUT / "appendix_figure_a3_seed_stability_revised2_summary.md"
    summary.write_text("\n".join([
        "# Appendix Figure A3 revised2 summary", "",
        "## Inputs", "",
        f"- `{RAW_CSV.relative_to(ROOT)}`", f"- `{STATS_CSV.relative_to(ROOT)}`", "",
        "## Rendering", "",
        "- Only Appendix Figure A3 was rendered; no other figure was modified.",
        "- Large/mean bars were removed.",
        "- Each panel uses individual-seed scatter points, a horizontal mean line, and a horizontal mean ± 1 std shaded band.",
        "- The three panels retain independent y-axes for PA-AUC, ASR@+500, and N80.", "",
        "## Outputs", "",
        f"- `{png.relative_to(ROOT)}`", f"- `{pdf.relative_to(ROOT)}`", "",
        "No training was started and no original CSV was modified.",
    ]) + "\n")


if __name__ == "__main__":
    main()
