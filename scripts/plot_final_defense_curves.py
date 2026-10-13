"""Create paper-ready ASR and clean-accuracy defense curves."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/saved_results/final_defense_60_150_r350_20260910"

def plot(column, title, ylabel, stem):
    data = pd.read_csv(BASE / "FINAL_curve_data_defense_asr.csv")
    figures = BASE / "figures"; figures.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for name, frame in data.groupby("defense", sort=False): axis.plot(frame["round"], frame[column], label=name, linewidth=1.8)
    axis.axvspan(60, 150, color="gray", alpha=.14, label="Attack phase")
    axis.set(title=title, xlabel="Global Round", ylabel=ylabel, xlim=(50,350)); axis.grid(alpha=.3); axis.legend()
    if column == "asr": axis.set_ylim(0,105)
    fig.tight_layout()
    for ext in ("png","pdf","svg"): fig.savefig(figures / f"{stem}.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)

if __name__ == "__main__":
    plot("asr", "ASR persistence under server-side defenses", "ASR (%)", "defense_asr_persistence")
    plot("clean_acc", "Clean accuracy under server-side defenses", "Clean Accuracy (%)", "defense_clean_accuracy")
