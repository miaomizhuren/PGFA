"""Export final PBU server-defense results after the four requested runs finish."""
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/saved_results/final_defense_60_150_r350_20260910"
STATUS_PATH = OUT / "DEFENSE_RUN_STATUS.csv"
KEY_ROUNDS = [59, 60, 150, 210, 270, 330, 350]
SOURCES = {
    "None": ROOT / "outputs/extended_attack_single_proto/ea01_s60_e150_r350/metrics_ea01_s60_e150_r350.db",
    "Norm Clipping": ROOT / "outputs/final_defense_60_150_r350/def01_norm_clipping_pbu_s60_e150_r350/metrics_def01_norm_clipping_pbu_s60_e150_r350.db",
    "Trimmed Mean": ROOT / "outputs/final_defense_60_150_r350/def02_trimmed_mean_pbu_s60_e150_r350/metrics_def02_trimmed_mean_pbu_s60_e150_r350.db",
    "Multi-Krum": ROOT / "outputs/final_defense_60_150_r350/def03_multi_krum_pbu_s60_e150_r350/metrics_def03_multi_krum_pbu_s60_e150_r350.db",
    "Light-FLAME": ROOT / "outputs/final_defense_60_150_r350/def04_flame_pbu_s60_e150_r350/metrics_def04_flame_pbu_s60_e150_r350.db",
}

def load(label, path):
    if not path.exists(): return pd.DataFrame()
    with sqlite3.connect(path) as conn: frame = pd.read_sql_query("select * from round_metrics order by round_index", conn)
    return frame.rename(columns={"round_index":"round", "clean_accuracy":"clean_acc", "attack_success_rate":"asr"}).assign(defense=label, source_file=str(path))

def metric(frame, round_, column):
    row = frame.loc[frame["round"].eq(round_), column]
    return None if row.empty else float(row.iloc[0])

def lifetime(frame, threshold):
    post = frame.loc[frame["round"].between(150, 350) & frame["asr"].notna()]
    failed = post.loc[post["asr"] < threshold, "round"]
    return ">200" if failed.empty else int(failed.iloc[0] - 150)

def status_by_label():
    mapping = {
        "def01_norm_clipping_pbu_s60_e150_r350": "Norm Clipping",
        "def02_trimmed_mean_pbu_s60_e150_r350": "Trimmed Mean",
        "def03_multi_krum_pbu_s60_e150_r350": "Multi-Krum",
        "def04_flame_pbu_s60_e150_r350": "Light-FLAME",
    }
    if not STATUS_PATH.exists(): return {}
    return {mapping.get(r.experiment): r.status for r in pd.read_csv(STATUS_PATH).itertuples() if mapping.get(r.experiment)}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    statuses, frames, rows, notes = status_by_label(), {}, [], []
    for label, path in SOURCES.items():
        frame = load(label, path); frames[label] = frame
        status = "COMPLETE" if label == "None" and not frame.empty and metric(frame,350,"clean_acc") is not None else statuses.get(label, "NOT_STARTED")
        row = {"Defense":label, "Status":status}
        if status == "COMPLETE":
            post = frame.loc[frame["round"].between(150,350) & frame["asr"].notna()]
            row.update({"CA@350":metric(frame,350,"clean_acc"), "ASR-0":metric(frame,150,"asr"), "ASR-60":metric(frame,210,"asr"), "ASR-120":metric(frame,270,"asr"), "ASR-180":metric(frame,330,"asr"), "ASR-200":metric(frame,350,"asr"), "L50":lifetime(frame,50), "L30":lifetime(frame,30), "PA-AUC_0:200":None if post.empty else float(np.trapezoid(post["asr"],post["round"])/200)})
        else:
            row.update({key:None for key in ["CA@350","ASR-0","ASR-60","ASR-120","ASR-180","ASR-200","L50","L30","PA-AUC_0:200"]})
            notes.append(f"{label}: {status}; round-350 summary metrics intentionally left blank.")
        rows.append(row)
        safe = label.replace(" ", "_").replace("-", "_")
        selected = frame.loc[frame["round"].isin(KEY_ROUNDS)].copy()
        selected = selected.set_index("round").reindex(KEY_ROUNDS).reset_index() if not selected.empty else pd.DataFrame({"round":KEY_ROUNDS})
        selected["relative_post_round"] = selected["round"] - 150
        selected["defense_name"] = label
        selected["run_status"] = status
        selected.to_csv(OUT / f"FINAL_key_rounds_{safe}.csv", index=False)
        frame.to_csv(OUT / f"FINAL_all_round_metrics_{safe}.csv", index=False)
    table = pd.DataFrame(rows); table.to_csv(OUT / "FINAL_TABLE_defense_persistence.csv", index=False)
    table = table.drop(columns="Status")  # Status remains in the CSV; the legacy LaTex layout omits it.
    curve = pd.concat([f for f in frames.values() if not f.empty], ignore_index=True)
    if not curve.empty:
        curve["relative_post_round"] = curve["round"] - 150
        curve.to_csv(OUT / "FINAL_curve_data_defense_asr.csv", index=False)
    latex = [r"\begin{table}[t]",r"\centering",r"\caption{Effectiveness of server-side defenses against PBU under the CIFAR-10 IID setting.}",r"\label{tab:defense_persistence}",r"\begin{tabular}{lccccccccc}",r"\toprule",r"Defense & CA@350 & ASR-0 & ASR-60 & ASR-120 & ASR-180 & ASR-200 & $L_{50}$ & $L_{30}$ & PA-AUC$_{0:200}$ \\",r"\midrule"]
    for _, row in table.iterrows(): latex.append(" & ".join(str(row.get(c,"")) if c in {"Defense","L50","L30"} else ("" if pd.isna(row.get(c)) else f"{row[c]:.2f}") for c in table.columns) + r" \\")
    latex += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (OUT / "FINAL_TABLE_defense_persistence_latex.tex").write_text("\n".join(latex), encoding="utf-8")
    (OUT / "FINAL_defense_diagnostics.csv").write_text("defense,round,selected_clients,selected_malicious_clients,multi_krum_selected_client_ids,multi_krum_rejected_client_ids,flame_kept_client_ids,flame_removed_client_ids,clip_norm,noise_std,source_log\n", encoding="utf-8")
    readme = "None imports the existing PBU ea01 result. COMPLETE rows contain final persistence metrics; partial or missing runs retain available curve/key-round data and blank round-350 summary metrics.\n"
    (OUT / "README_where_to_open.txt").write_text(readme + ("\n".join(notes) + "\n" if notes else ""), encoding="utf-8")

if __name__ == "__main__": main()
