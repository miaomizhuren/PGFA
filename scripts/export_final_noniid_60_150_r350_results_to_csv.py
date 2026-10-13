"""Export IID reference and completed final Non-IID persistence experiments."""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "outputs/saved_results/final_noniid_60_150_r350_20260912"
KEY = [59, 60, 150, 210, 270, 330, 350]
ATTACK_END = 150
EXPERIMENTS = [
    ("IID", "CE-UAP", [ROOT / "outputs/final_ablation_60_150_r350/abl01_ce_only_s60_e150_r350", ROOT / "outputs/final_main_60_150_r350/ce_uap_s60_e150_r350"]),
    ("IID", "PBU", [ROOT / "outputs/extended_attack_single_proto/ea01_s60_e150_r350", ROOT / "outputs/saved_results/extended_attack_single_proto_ea01_20260909", ROOT / "outputs/mechanism_feature_margin_60_150_r350/mech_pbu_s60_e150_r350_saveckpt"]),
    ("Dirichlet alpha=1.0", "CE-UAP", [ROOT / "outputs/final_noniid_60_150_r350/noniid_a10_ce_uap_s60_e150_r350"]),
    ("Dirichlet alpha=1.0", "PBU", [ROOT / "outputs/final_noniid_60_150_r350/noniid_a10_pbu_s60_e150_r350"]),
    ("Dirichlet alpha=0.5", "CE-UAP", [ROOT / "outputs/final_noniid_60_150_r350/noniid_a05_ce_uap_s60_e150_r350"]),
    ("Dirichlet alpha=0.5", "PBU", [ROOT / "outputs/final_noniid_60_150_r350/noniid_a05_pbu_s60_e150_r350"]),
]
ROUND_COLUMNS = ["round", "relative_post_round", "partition", "method", "clean_acc", "asr", "attack_active", "selected_malicious_clients", "num_selected_malicious", "num_attack_samples", "num_successful_attacks", "psnr", "ssim", "linf", "source_file"]

def find_db(candidates):
    for directory in candidates:
        if directory.is_dir():
            found = sorted(directory.rglob("metrics*.db"))
            if found:
                return found[0]
    return None

def load_db(path):
    if path is None:
        return None
    with sqlite3.connect(path) as conn:
        frame = pd.read_sql_query("SELECT * FROM round_metrics ORDER BY round_index", conn)
    frame = frame.rename(columns={"round_index": "round", "clean_accuracy": "clean_acc", "attack_success_rate": "asr"})
    frame["round"] = pd.to_numeric(frame["round"], errors="coerce")
    return frame.dropna(subset=["round"]).assign(round=lambda x: x["round"].astype(int))

def at(frame, round_index, column):
    if frame is None or column not in frame:
        return None
    values = frame.loc[frame["round"] == round_index, column]
    return None if values.empty else values.iloc[-1]

def first_below(frame, threshold):
    if frame is None:
        return None
    post = frame.loc[(frame["round"] >= ATTACK_END) & frame["asr"].notna()].sort_values("round")
    hits = post.loc[post["asr"] < threshold, "round"]
    return ">200" if hits.empty else int(hits.iloc[0] - ATTACK_END)

def auc(frame):
    if frame is None:
        return None
    post = frame.loc[(frame["round"] >= ATTACK_END) & (frame["round"] <= 350) & frame["asr"].notna(), ["round", "asr"]].drop_duplicates("round", keep="last").sort_values("round")
    if post.empty or post.iloc[0]["round"] != ATTACK_END or post.iloc[-1]["round"] != 350:
        return None
    return float(np.trapezoid(post["asr"].to_numpy(float), post["round"].to_numpy(float)) / 200.0)

def tag(partition, method):
    return f"{partition.replace('Dirichlet alpha=', 'Dirichlet_alpha').replace('=', '').replace(' ', '_')}_{method}"

def table_row(partition, method, frame):
    return {"Partition": partition, "Method": method, "CA@350": at(frame, 350, "clean_acc"), "ASR-0": at(frame, 150, "asr"), "ASR-60": at(frame, 210, "asr"), "ASR-120": at(frame, 270, "asr"), "ASR-180": at(frame, 330, "asr"), "ASR-200": at(frame, 350, "asr"), "L50": first_below(frame, 50), "L30": first_below(frame, 30), "PA-AUC_0:200": auc(frame)}

def latex_value(value):
    return "--" if value is None or pd.isna(value) else (f"{value:.2f}" if isinstance(value, (int, float, np.number)) else str(value))

def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    frames, sources, rows = {}, {}, []
    for partition, method, candidates in EXPERIMENTS:
        key = (partition, method); sources[key] = find_db(candidates); frames[key] = load_db(sources[key]); rows.append(table_row(partition, method, frames[key]))
    table = pd.DataFrame(rows)
    for partition in table.Partition.unique():
        ce = table[(table.Partition == partition) & (table.Method == "CE-UAP")].iloc[0]
        pbu_i = table.index[(table.Partition == partition) & (table.Method == "PBU")][0]
        table.loc[pbu_i, "PBU_minus_CE_PA_AUC"] = table.loc[pbu_i, "PA-AUC_0:200"] - ce["PA-AUC_0:200"] if pd.notna(table.loc[pbu_i, "PA-AUC_0:200"]) and pd.notna(ce["PA-AUC_0:200"]) else None
        table.loc[pbu_i, "PBU_minus_CE_ASR_200"] = table.loc[pbu_i, "ASR-200"] - ce["ASR-200"] if pd.notna(table.loc[pbu_i, "ASR-200"]) and pd.notna(ce["ASR-200"]) else None
    table.to_csv(SAVE / "FINAL_TABLE_noniid_persistence.csv", index=False, encoding="utf-8-sig")
    cols = ["CA@350", "ASR-0", "ASR-60", "ASR-120", "ASR-180", "ASR-200", "L50", "L30", "PA-AUC_0:200"]
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Impact of data heterogeneity on the persistence of CE-UAP and PBU.}", r"\label{tab:noniid_persistence}", r"\begin{tabular}{llccccccccc}", r"\toprule", r"Partition & Method & CA@350 & ASR-0 & ASR-60 & ASR-120 & ASR-180 & ASR-200 & $L_{50}$ & $L_{30}$ & PA-AUC$_{0:200}$ \\", r"\midrule"]
    for _, row in table.iterrows():
        partition = row.Partition.replace("alpha=", r"$\alpha=").replace(".0", ".0$").replace(".5", ".5$") if row.Partition.startswith("Dirichlet") else row.Partition
        lines.append(partition + " & " + row.Method + " & " + " & ".join(latex_value(row[c]) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (SAVE / "FINAL_TABLE_noniid_persistence_latex.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    curves = []
    for partition, method, _ in EXPERIMENTS:
        frame, source = frames[(partition, method)], sources[(partition, method)]; name = tag(partition, method)
        if frame is None:
            full = pd.DataFrame(columns=ROUND_COLUMNS); key_frame = full
        else:
            full = frame.copy(); full["relative_post_round"] = full["round"] - ATTACK_END; full["partition"] = partition; full["method"] = method; full["source_file"] = str(source)
            for column in ROUND_COLUMNS:
                if column not in full: full[column] = None
            full = full[ROUND_COLUMNS]; key_frame = full[full["round"].isin(KEY)]
            curves.append(full[["round", "relative_post_round", "partition", "method", "clean_acc", "asr", "attack_active", "source_file"]])
        full.to_csv(SAVE / f"FINAL_all_round_metrics_{name}.csv", index=False, encoding="utf-8-sig"); key_frame.to_csv(SAVE / f"FINAL_key_rounds_{name}.csv", index=False, encoding="utf-8-sig")
    pd.concat(curves, ignore_index=True).to_csv(SAVE / "FINAL_curve_data_noniid_asr.csv", index=False, encoding="utf-8-sig") if curves else pd.DataFrame(columns=["round", "relative_post_round", "partition", "method", "clean_acc", "asr", "attack_active", "source_file"]).to_csv(SAVE / "FINAL_curve_data_noniid_asr.csv", index=False, encoding="utf-8-sig")
    readme = ["Main table: " + str(SAVE / "FINAL_TABLE_noniid_persistence.csv"), "LaTex table: " + str(SAVE / "FINAL_TABLE_noniid_persistence_latex.tex"), "Curve data: " + str(SAVE / "FINAL_curve_data_noniid_asr.csv"), "", "IID rows are imported from existing results and are not rerun.", "The four Dirichlet rows are produced by this new run after completion.", "alpha=0.1 was not run: extreme heterogeneity can make training unstable; this study focuses on moderate and strong heterogeneity."]
    (SAVE / "README_where_to_open.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Exported Non-IID persistence artifacts to {SAVE}")
if __name__ == "__main__": main()
