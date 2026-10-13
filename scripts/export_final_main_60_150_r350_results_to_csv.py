"""Export the final 60--150/350 persistence table from completed local runs."""
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SAVE = ROOT / "outputs/saved_results/final_main_60_150_r350_20260909"
KEY = [59, 60, 150, 210, 270, 330, 350]
ATTACK_END = 150
METHODS = ["DBA", "Neurotoxin-UAP", "CE-UAP", "PBU"]
BASELINE_DIRS = {
    "DBA": ROOT / "outputs/final_main_60_150_r350/dba_random_size5_pr06_w1_s60_e150_r350",
    "Neurotoxin-UAP": ROOT / "outputs/final_main_60_150_r350/neurotoxin_uap_mr005_gradient_s60_e150_r350",
    "CE-UAP": ROOT / "outputs/final_main_60_150_r350/ce_uap_s60_e150_r350",
}
PBU_CANDIDATES = [
    ROOT / "outputs/extended_attack_single_proto/ea01_s60_e150_r350",
    ROOT / "outputs/saved_results/extended_attack_single_proto_ea01_20260909",
    ROOT / "outputs/final_best",
    ROOT / "outputs/reproduce_best",
]
ROUND_COLUMNS = [
    "round", "relative_post_round", "clean_acc", "asr", "attack_active", "selected_clients",
    "selected_malicious_clients", "num_selected_malicious", "num_attack_samples",
    "num_successful_attacks", "psnr", "ssim", "linf", "source_file",
]


def metric_db_candidates(directory):
    if not directory.is_dir():
        return []
    return sorted(directory.rglob("metrics*.db"))


def usable_database(path):
    try:
        with sqlite3.connect(path) as connection:
            frame = pd.read_sql_query("SELECT round_index, attack_success_rate FROM round_metrics", connection)
        rounds = pd.to_numeric(frame["round_index"], errors="coerce")
        return {59, 60, 150, 350}.issubset(set(rounds.dropna().astype(int)))
    except (sqlite3.Error, pd.errors.DatabaseError):
        return False


def find_pbu_database():
    for directory in PBU_CANDIDATES:
        for path in metric_db_candidates(directory):
            if usable_database(path):
                return path
    for path in ROOT.glob("outputs/**/metrics*.db"):
        name = str(path).lower()
        if ("s60_e150_r350" in name or "ea01" in name) and usable_database(path):
            return path
    return None


def find_baseline_database(method):
    for path in metric_db_candidates(BASELINE_DIRS[method]):
        if usable_database(path):
            return path
    return None


def load_database(path):
    with sqlite3.connect(path) as connection:
        frame = pd.read_sql_query("SELECT * FROM round_metrics ORDER BY round_index", connection)
    frame = frame.rename(columns={
        "round_index": "round", "clean_accuracy": "clean_acc",
        "attack_success_rate": "asr",
    })
    frame["round"] = pd.to_numeric(frame["round"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["round"]).copy()
    frame["round"] = frame["round"].astype(int)
    return frame


def value_at(frame, round_index, column):
    values = frame.loc[frame["round"] == round_index, column]
    return None if values.empty else values.iloc[-1]


def first_below(frame, threshold):
    post = frame.loc[(frame["round"] >= ATTACK_END) & frame["asr"].notna()].sort_values("round")
    hits = post.loc[post["asr"] < threshold, "round"]
    return ">200" if hits.empty else int(hits.iloc[0] - ATTACK_END)


def normalized_auc(frame):
    post = frame.loc[(frame["round"] >= ATTACK_END) & (frame["round"] <= 350) & frame["asr"].notna(), ["round", "asr"]]
    post = post.drop_duplicates(subset=["round"], keep="last").sort_values("round")
    if post.empty or post["round"].iloc[0] != ATTACK_END or post["round"].iloc[-1] != 350:
        return None
    return float(np.trapz(post["asr"].to_numpy(dtype=float), post["round"].to_numpy(dtype=float)) / 200.0)


def empty_key_frame():
    return pd.DataFrame({"round": KEY, "relative_post_round": [round_index - ATTACK_END for round_index in KEY]})


def key_frame(frame, source):
    if frame is None:
        result = empty_key_frame()
    else:
        result = frame.drop_duplicates(subset=["round"], keep="last").set_index("round").reindex(KEY).rename_axis("round").reset_index()
        result["relative_post_round"] = result["round"] - ATTACK_END
    for column in ROUND_COLUMNS:
        if column not in result:
            result[column] = None
    result["source_file"] = str(source) if source else None
    return result[ROUND_COLUMNS]


def table_row(method, frame, source):
    if frame is None:
        return {"Method": method, **{column: None for column in ["CA@350", "ASR-0", "ASR-60", "ASR-120", "ASR-180", "ASR-200", "L50", "L30", "PA-AUC_0:200"]}}
    return {
        "Method": method, "CA@350": value_at(frame, 350, "clean_acc"),
        "ASR-0": value_at(frame, 150, "asr"), "ASR-60": value_at(frame, 210, "asr"),
        "ASR-120": value_at(frame, 270, "asr"), "ASR-180": value_at(frame, 330, "asr"),
        "ASR-200": value_at(frame, 350, "asr"), "L50": first_below(frame, 50),
        "L30": first_below(frame, 30), "PA-AUC_0:200": normalized_auc(frame),
    }


def latex_value(value):
    if value is None or pd.isna(value):
        return "--"
    return f"{value:.2f}" if isinstance(value, (int, float, np.number)) else str(value)


def write_latex(table):
    columns = ["CA@350", "ASR-0", "ASR-60", "ASR-120", "ASR-180", "ASR-200", "L50", "L30", "PA-AUC_0:200"]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{CIFAR-10 IID setting: attack effectiveness and post-attack persistence of different methods.}",
        r"\label{tab:main_persistence}", r"\begin{tabular}{lccccccccc}", r"\toprule",
        r"Method & CA@350 & ASR-0 & ASR-60 & ASR-120 & ASR-180 & ASR-200 & $L_{50}$ & $L_{30}$ & PA-AUC$_{0:200}$ \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        method = r"\textbf{PBU}" if row["Method"] == "PBU" else row["Method"]
        lines.append(method + " & " + " & ".join(latex_value(row[column]) for column in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (SAVE / "FINAL_TABLE_main_persistence_latex.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dba_random_participation(frame):
    """Summarize actual malicious selection during the inclusive attack window."""
    columns = [
        "attack_start", "attack_end", "num_attack_rounds", "total_selected_malicious_count",
        "avg_selected_malicious_per_round", "rounds_with_0_malicious", "rounds_with_1_malicious",
        "rounds_with_2_malicious", "rounds_with_3_malicious", "effective_poisoning_rounds",
        "effective_poisoning_round_ratio",
    ]
    output = SAVE / "DBA_random_participation_summary.csv"
    if frame is None:
        pd.DataFrame([{column: None for column in columns}]).to_csv(output, index=False, encoding="utf-8-sig")
        return
    attack = frame.loc[(frame["round"] >= 60) & (frame["round"] <= ATTACK_END)].copy()
    counts = pd.to_numeric(attack["num_selected_malicious"], errors="coerce").fillna(0).astype(int)
    total_rounds = len(attack)
    total_selected = int(counts.sum())
    effective = int((counts > 0).sum())
    row = {
        "attack_start": 60, "attack_end": ATTACK_END, "num_attack_rounds": total_rounds,
        "total_selected_malicious_count": total_selected,
        "avg_selected_malicious_per_round": total_selected / total_rounds if total_rounds else None,
        "rounds_with_0_malicious": int((counts == 0).sum()), "rounds_with_1_malicious": int((counts == 1).sum()),
        "rounds_with_2_malicious": int((counts == 2).sum()), "rounds_with_3_malicious": int((counts == 3).sum()),
        "effective_poisoning_rounds": effective,
        "effective_poisoning_round_ratio": effective / total_rounds if total_rounds else None,
    }
    pd.DataFrame([row], columns=columns).to_csv(output, index=False, encoding="utf-8-sig")


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    sources = {method: find_baseline_database(method) for method in BASELINE_DIRS}
    sources["PBU"] = find_pbu_database()
    frames = {method: (load_database(path) if path else None) for method, path in sources.items()}
    table = pd.DataFrame([table_row(method, frames[method], sources[method]) for method in METHODS])
    table.to_csv(SAVE / "FINAL_TABLE_main_persistence.csv", index=False, encoding="utf-8-sig")
    write_latex(table)
    write_dba_random_participation(frames["DBA"])
    curves = []
    for method in METHODS:
        frame, source = frames[method], sources[method]
        key_frame(frame, source).to_csv(SAVE / f"FINAL_key_rounds_{method}.csv", index=False, encoding="utf-8-sig")
        if frame is None:
            pd.DataFrame(columns=ROUND_COLUMNS).to_csv(SAVE / f"FINAL_all_round_metrics_{method}.csv", index=False, encoding="utf-8-sig")
            continue
        frame = frame.copy(); frame["relative_post_round"] = frame["round"] - ATTACK_END; frame["source_file"] = str(source)
        for column in ROUND_COLUMNS:
            if column not in frame: frame[column] = None
        frame[ROUND_COLUMNS].to_csv(SAVE / f"FINAL_all_round_metrics_{method}.csv", index=False, encoding="utf-8-sig")
        curve = frame.loc[frame["round"] >= 50, ["round", "relative_post_round", "clean_acc", "asr", "attack_active", "source_file"]].copy()
        curve.insert(2, "method", method); curves.append(curve)
    curve_output = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=["round", "relative_post_round", "method", "clean_acc", "asr", "attack_active", "source_file"])
    curve_output.to_csv(SAVE / "FINAL_curve_data_main_asr.csv", index=False, encoding="utf-8-sig")
    readme = [f"Main table: {SAVE / 'FINAL_TABLE_main_persistence.csv'}", f"Curve data: {SAVE / 'FINAL_curve_data_main_asr.csv'}", "", "Sources:"]
    for method in METHODS:
        status = str(sources[method]) if sources[method] else "missing (no value filled)"
        readme.append(f"- {method}: {status}")
    readme.extend([
        "", "PBU is imported from the existing source above and is not rerun.",
        "DBA source uses random client participation; PBU uses the high-participation malicious-client setting. State this protocol difference explicitly in the paper.",
        "Neurotoxin-UAP and CE-UAP sources are this run's baseline output directories once training is complete.",
    ])
    (SAVE / "README_where_to_open.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Exported final table to {SAVE}")


if __name__ == "__main__":
    main()
