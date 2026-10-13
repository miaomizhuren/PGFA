"""Create publication-ready persistence and clean-accuracy ablation plots."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/saved_results/final_ablation_60_150_r350_20260911"
DATA = OUT / "FINAL_curve_data_ablation_asr.csv"

def draw(metric, title, ylabel, stem):
    frame = pd.read_csv(DATA)
    if frame.empty:
        raise RuntimeError("No curve data found; run the export script after experiments finish.")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for variant in ["CE-only", "Delta-only", "Train-only", "Full PBU"]:
        series = frame.loc[frame["variant"].eq(variant)].dropna(subset=[metric])
        if not series.empty:
            ax.plot(series["round"], series[metric], linewidth=2, label=variant)
    ax.axvspan(60, 150, color="grey", alpha=0.16, label="Attack phase")
    ax.set(title=title, xlabel="Global Round", ylabel=ylabel, xlim=(50, 350), ylim=(0, 105))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(fig_dir / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(fig)

def main():
    draw("asr", "Ablation study of prototype guidance", "ASR (%)", "ablation_asr_persistence")
    draw("clean_acc", "Clean accuracy in ablation study", "Clean Accuracy (%)", "ablation_clean_accuracy")

if __name__ == "__main__":
    main()
