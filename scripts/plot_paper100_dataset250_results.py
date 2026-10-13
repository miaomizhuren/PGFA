#!/usr/bin/env python3
"""Plot completed dataset250 metrics; incomplete rows are skipped safely."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SUMMARY=ROOT/"experiments/summaries/paper100_dataset250/paper100_dataset250_summary_all.csv"
FIG=ROOT/"experiments/figures/paper100_dataset250"
ROUNDS=[110,150,180,200,240,250]

def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png","pdf"):
        fig.savefig(FIG/f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_curve(data, metric, filename, ylabel):
    fig, ax=plt.subplots(figsize=(7,4))
    for _, row in data.iterrows():
        values=[pd.to_numeric(row.get(f"{metric}@{r}"), errors="coerce") for r in ROUNDS]
        if any(pd.notna(values)):
            ax.plot(ROUNDS, values, marker="o", label=f"{row['dataset']} {row['method']}")
    ax.set(xlabel="Global Round", ylabel=ylabel)
    ax.grid(alpha=.3)
    if ax.lines: ax.legend()
    save(fig, filename)

def main():
    if not SUMMARY.is_file():
        print(f"Missing summary CSV: {SUMMARY}"); return 0
    data=pd.read_csv(SUMMARY)
    plot_curve(data, "ASR", "dataset250_asr_curve", "ASR (%)")
    plot_curve(data, "CleanACC", "dataset250_cleanacc_curve", "Clean Accuracy (%)")
    fig, ax=plt.subplots(figsize=(7,4))
    labels=[]; values=[]
    for _, row in data.iterrows():
        value=pd.to_numeric(row.get("AUC@110-250"), errors="coerce")
        if pd.notna(value): labels.append(f"{row['dataset']}\n{row['method']}"); values.append(value)
    ax.bar(labels, values, color=["#4c78a8","#f58518","#54a24b","#e45756"][:len(values)])
    ax.set(ylabel="PA-AUC (110–250)", title="Dataset generalization")
    ax.grid(axis="y", alpha=.3)
    save(fig, "dataset250_auc_bar")
    print(FIG)
if __name__=="__main__":
    main()

