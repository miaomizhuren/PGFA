"""Export completed Non-IID multi-prototype results to fixed comparison CSVs."""

from pathlib import Path
import re
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "outputs/saved_results/non_iid_multi_proto_20260908"
KEY_ROUNDS = [110, 111, 120, 130, 150, 180, 200, 240, 250]
EXPERIMENTS = {
    "alpha05": ROOT / "outputs/non_iid_multi_proto/non_iid_alpha05_multi_proto_s60_e110_seed2026",
    "alpha01": ROOT / "outputs/non_iid_multi_proto/non_iid_alpha01_multi_proto_s60_e110_seed2026",
}


def normalize(df):
    aliases = {"round_index": "round", "round_id": "round", "clean_accuracy": "clean_acc",
               "cleanacc": "clean_acc", "attack_success_rate": "asr", "attack_success": "asr"}
    names = []
    for name in df.columns:
        key = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
        names.append(aliases.get(key, key))
    df = df.copy(); df.columns = names
    return df


def find_metrics(base):
    candidates = []
    for path in [*base.rglob("*.db"), *base.rglob("*.sqlite"), *base.rglob("*.sqlite3")]:
        try:
            with sqlite3.connect(path) as conn:
                for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                    frame = normalize(pd.read_sql_query(f'SELECT * FROM "{table}"', conn))
                    score = 10 * ("round" in frame) + 10 * ("asr" in frame) + 8 * ("clean_acc" in frame)
                    if score >= 20:
                        candidates.append((score, "sqlite", path, table, frame))
        except Exception as exc:
            print(f"[WARN] unable to read {path}: {exc}")
    for path in base.rglob("*.csv"):
        if path.name.startswith("FINAL_"):
            continue
        try:
            frame = normalize(pd.read_csv(path))
            score = 10 * ("round" in frame) + 10 * ("asr" in frame) + 8 * ("clean_acc" in frame)
            if score >= 20:
                candidates.append((score, "csv", path, "", frame))
        except Exception as exc:
            print(f"[WARN] unable to read {path}: {exc}")
    if not candidates:
        raise FileNotFoundError(f"No completed round/ASR metrics found under {base}")
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0]


def tidy(frame):
    frame = frame.copy()
    frame["round"] = pd.to_numeric(frame["round"], errors="coerce")
    frame = frame.dropna(subset=["round"]); frame["round"] = frame["round"].astype(int)
    for column in ("clean_acc", "asr", "psnr", "ssim", "linf"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    preferred = ["round", "clean_acc", "asr", "selected_clients", "selected_malicious_clients",
                 "attack_active", "average_train_loss", "psnr", "ssim", "linf",
                 "num_attack_samples", "num_successful_attacks"]
    return frame[[c for c in preferred if c in frame] + [c for c in frame if c not in preferred]].sort_values("round")


SAVE_DIR.mkdir(parents=True, exist_ok=True)
results, sources = {}, {}
for tag, directory in EXPERIMENTS.items():
    _, kind, path, table, frame = find_metrics(directory)
    frame = tidy(frame)
    frame.to_csv(SAVE_DIR / f"FINAL_all_round_metrics_{tag}.csv", index=False, encoding="utf-8-sig")
    frame[frame["round"].isin(KEY_ROUNDS)].to_csv(
        SAVE_DIR / f"FINAL_key_rounds_{tag}.csv", index=False, encoding="utf-8-sig"
    )
    results[tag] = frame; sources[tag] = (kind, path, table)

rows = []
for tag, alpha in (("alpha05", 0.5), ("alpha01", 0.1)):
    frame = results[tag].set_index("round")
    row = {"alpha": alpha}
    for round_index in KEY_ROUNDS:
        row[f"ASR@{round_index}"] = frame.at[round_index, "asr"] if round_index in frame.index else None
    for column, label in (("clean_acc", "CleanACC@250"), ("psnr", "PSNR"), ("ssim", "SSIM"), ("linf", "Linf")):
        row[label] = frame.at[250, column] if column in frame and 250 in frame.index else None
    rows.append(row)
compare_path = SAVE_DIR / "FINAL_compare_alpha05_alpha01.csv"
pd.DataFrame(rows).to_csv(compare_path, index=False, encoding="utf-8-sig")

readme = ["Non-IID multi-prototype 结果已整理。", "", "优先打开横向对比：", str(compare_path), ""]
for tag in ("alpha05", "alpha01"):
    kind, path, table = sources[tag]
    readme += [f"{tag} 关键轮次：", str(SAVE_DIR / f"FINAL_key_rounds_{tag}.csv"),
               f"source: {kind} {path}" + (f" table={table}" if table else ""), ""]
(SAVE_DIR / "README_where_to_open.txt").write_text("\n".join(readme), encoding="utf-8")
print(f"Exported comparison: {compare_path}")
