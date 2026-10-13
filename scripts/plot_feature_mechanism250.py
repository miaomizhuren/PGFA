#!/usr/bin/env python3
"""Plot feature-space mechanism250 results from the analysis CSV."""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "experiments/summaries/paper100_mechanism250/feature_mechanism250_summary.csv"
FIG_DIR = ROOT / "experiments/figures/paper100_mechanism250"

def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", dpi=180)
    plt.close(fig)

def main():
    if not CSV_PATH.is_file():
        print(f"Missing analysis CSV: {CSV_PATH}. Run analyze_feature_mechanism250.py first.", file=sys.stderr)
        return 1
    data = pd.read_csv(CSV_PATH)
    if data.empty:
        print(f"Analysis CSV has no rows: {CSV_PATH}", file=sys.stderr)
        return 1
    grouped = data.groupby(["experiment_id", "round"], as_index=False).mean(numeric_only=True)
    for column, name, ylabel in (("feature_margin", "feature_margin", "Feature margin"), ("target_sim", "target_similarity", "Target similarity")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for experiment, subset in grouped.groupby("experiment_id"):
            ax.plot(subset["round"], subset[column], marker="o", label=experiment)
        ax.set(xlabel="Round", ylabel=ylabel); ax.legend(fontsize=8); ax.grid(alpha=.3)
        save(fig, name)
    fig, ax = plt.subplots(figsize=(6, 4))
    for experiment, subset in grouped.groupby("experiment_id"):
        ax.scatter(subset["feature_margin"], subset["asr"], label=experiment)
    ax.set(xlabel="Feature margin", ylabel="ASR (%)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, "margin_asr_relation")
    print(f"Wrote figures under {FIG_DIR}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
