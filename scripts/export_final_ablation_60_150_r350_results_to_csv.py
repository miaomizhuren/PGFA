"""Export final prototype-guidance ablation results without fabricating missing rounds."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/saved_results/final_ablation_60_150_r350_20260911"
KEYS = [59, 60, 150, 210, 270, 330, 350]
SOURCES = {
    "CE-only": ROOT / "outputs/final_ablation_60_150_r350/abl01_ce_only_s60_e150_r350/metrics_abl01_ce_only_s60_e150_r350.db",
    "Delta-only": ROOT / "outputs/final_ablation_60_150_r350/abl02_delta_only_s60_e150_r350/metrics_abl02_delta_only_s60_e150_r350.db",
    "Train-only": ROOT / "outputs/final_ablation_60_150_r350/abl03_train_only_s60_e150_r350/metrics_abl03_train_only_s60_e150_r350.db",
    "Full PBU": ROOT / "outputs/extended_attack_single_proto/ea01_s60_e150_r350/metrics_ea01_s60_e150_r350.db",
}
SUMMARY_COLUMNS = ["Variant", "CA@350", "ASR-0", "ASR-60", "ASR-120", "ASR-180", "ASR-200", "L50", "L30", "PA-AUC_0:200"]

def load(path):
    if not path.exists():
        return pd.DataFrame()
    with sqlite3.connect(path) as conn:
        frame = pd.read_sql_query("select * from round_metrics order by round_index", conn)
    return frame.rename(columns={"round_index": "round", "clean_accuracy": "clean_acc", "attack_success_rate": "asr"})

def value(frame, round_, column):
    rows = frame.loc[frame["round"].eq(round_), column]
    return None if rows.empty else float(rows.iloc[0])

def lifetime(frame, threshold):
    post = frame.loc[frame["round"].between(150, 350) & frame["asr"].notna()]
    below = post.loc[post["asr"] < threshold, "round"]
    return ">200" if below.empty else int(below.iloc[0] - 150)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, summaries, readme = {}, [], []
    for variant, source in SOURCES.items():
        frame = load(source)
        frames[variant] = frame
        complete = not frame.empty and value(frame, 350, "clean_acc") is not None
        if complete:
            post = frame.loc[frame["round"].between(150, 350) & frame["asr"].notna()]
            summaries.append({"Variant": variant, "CA@350": value(frame, 350, "clean_acc"), "ASR-0": value(frame, 150, "asr"), "ASR-60": value(frame, 210, "asr"), "ASR-120": value(frame, 270, "asr"), "ASR-180": value(frame, 330, "asr"), "ASR-200": value(frame, 350, "asr"), "L50": lifetime(frame, 50), "L30": lifetime(frame, 30), "PA-AUC_0:200": float(np.trapezoid(post["asr"], post["round"]) / 200) if not post.empty else None})
        else:
            summaries.append({"Variant": variant})
            readme.append(f"- {variant}: missing or partial source DB; final summary metrics are blank.")
        selected = frame.loc[frame["round"].isin(KEYS)].copy() if not frame.empty else pd.DataFrame()
        selected = selected.set_index("round").reindex(KEYS).reset_index() if not selected.empty else pd.DataFrame({"round": KEYS})
        selected["relative_post_round"] = selected["round"] - 150
        selected["variant"] = variant
        selected["source_file"] = str(source)
        selected = selected.reindex(columns=["round", "relative_post_round", "clean_acc", "asr", "attack_active", "selected_malicious_clients", "num_selected_malicious", "num_attack_samples", "num_successful_attacks", "psnr", "ssim", "linf", "variant", "source_file"])
        selected.to_csv(OUT / f"FINAL_key_rounds_{variant.replace(' ', '-')}.csv", index=False)
        all_metrics = frame.copy()
        if all_metrics.empty:
            all_metrics = pd.DataFrame(columns=["round", "clean_acc", "asr", "attack_active"])
        all_metrics["variant"], all_metrics["source_file"] = variant, str(source)
        all_metrics.to_csv(OUT / f"FINAL_all_round_metrics_{variant.replace(' ', '-')}.csv", index=False)
        readme.append(f"- {variant}: {source}")
    table = pd.DataFrame(summaries).reindex(columns=SUMMARY_COLUMNS)
    table.to_csv(OUT / "FINAL_TABLE_ablation_persistence.csv", index=False)
    latex = [r"\begin{table}[t]", r"\centering", r"\caption{Ablation study of prototype guidance in UAP generation and malicious training.}", r"\label{tab:ablation_pbu}", r"\begin{tabular}{lccccccccc}", r"\toprule", r"Variant & CA@350 & ASR-0 & ASR-60 & ASR-120 & ASR-180 & ASR-200 & $L_{50}$ & $L_{30}$ & PA-AUC$_{0:200}$ \\", r"\midrule"]
    for _, row in table.iterrows():
        latex.append(" & ".join(str(row[c]) if c in {"Variant", "L50", "L30"} else ("" if pd.isna(row[c]) else f"{row[c]:.2f}") for c in SUMMARY_COLUMNS) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "FINAL_TABLE_ablation_persistence_latex.tex").write_text("\n".join(latex), encoding="utf-8")
    curves = []
    for variant, frame in frames.items():
        if not frame.empty:
            curves.append(frame.assign(relative_post_round=frame["round"] - 150, variant=variant, source_file=str(SOURCES[variant])).reindex(columns=["round", "relative_post_round", "variant", "clean_acc", "asr", "attack_active", "source_file"]))
    pd.concat(curves, ignore_index=True).to_csv(OUT / "FINAL_curve_data_ablation_asr.csv", index=False) if curves else pd.DataFrame(columns=["round", "relative_post_round", "variant", "clean_acc", "asr", "attack_active", "source_file"]).to_csv(OUT / "FINAL_curve_data_ablation_asr.csv", index=False)
    (OUT / "README_where_to_open.txt").write_text("\n".join(["Ablation outputs:", "- Main table: FINAL_TABLE_ablation_persistence.csv", "- LaTex table: FINAL_TABLE_ablation_persistence_latex.tex", "- Curve data: FINAL_curve_data_ablation_asr.csv", "- Full PBU is imported from an existing result and is not rerun.", "", "Source DBs:"] + readme) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
