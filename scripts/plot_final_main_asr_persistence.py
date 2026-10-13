"""Plot the final-main ASR persistence curves from the exported CSV."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "outputs/saved_results/final_main_60_150_r350_20260909"
INPUT = SAVE / "FINAL_curve_data_main_asr.csv"
FIGURES = SAVE / "figures"
LABELS = {"DBA": "DBA", "Neurotoxin-UAP": "Neurotoxin-UAP", "CE-UAP": "CE-UAP", "PBU": "PBU (Ours)"}


def main():
    if not INPUT.is_file():
        raise SystemExit(f"Missing curve CSV: {INPUT}; run the exporter after training first.")
    frame = pd.read_csv(INPUT)
    if frame.empty:
        raise SystemExit("Curve CSV contains no completed method data.")
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 14, "legend.fontsize": 10})
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    axis.axvspan(60, 150, color="tab:red", alpha=0.12, label="Attack stage")
    for method in ("DBA", "Neurotoxin-UAP", "CE-UAP", "PBU"):
        subset = frame.loc[(frame["method"] == method) & frame["asr"].notna()].sort_values("round")
        if not subset.empty:
            axis.plot(subset["round"], subset["asr"], linewidth=2.0, label=LABELS[method])
    axis.set_title("Main ASR persistence")
    axis.set_xlabel("Global Round")
    axis.set_ylabel("ASR (%)")
    axis.set_xlim(50, 350)
    axis.set_ylim(0, 105)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, frameon=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(FIGURES / f"main_asr_persistence.{extension}", dpi=300 if extension == "png" else None)
    plt.close(figure)


if __name__ == "__main__":
    main()
