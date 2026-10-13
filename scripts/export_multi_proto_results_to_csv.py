"""Export completed multi-prototype experiment metrics to fixed, VS Code-friendly CSVs."""

from pathlib import Path
import re
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "outputs/saved_results/multi_proto_aggregation_20260908"
EXTRA_DIR = ROOT / "outputs/reproduce_best/reproduce_pbu_ok_equiv_multi_proto_s60_e110_seed2026"
KEY_ROUNDS = [110, 111, 120, 130, 150, 180, 200, 240, 250]


def normalized(df):
    aliases = {"round_index": "round", "clean_accuracy": "clean_acc",
               "cleanacc": "clean_acc", "attack_success_rate": "asr",
               "attack_success": "asr", "backdoor_asr": "asr"}
    cols = []
    for column in df.columns:
        key = re.sub(r"[^a-z0-9]+", "_", str(column).lower()).strip("_")
        cols.append(aliases.get(key, key))
    df = df.copy(); df.columns = cols
    return df


def score(df):
    return 10 * ("round" in df) + 10 * ("asr" in df) + 8 * ("clean_acc" in df)


SAVE_DIR.mkdir(parents=True, exist_ok=True)
candidates = []
for base in (SAVE_DIR, EXTRA_DIR):
    if not base.exists():
        continue
    for path in [*base.rglob("*.db"), *base.rglob("*.sqlite"), *base.rglob("*.sqlite3")]:
        try:
            with sqlite3.connect(path) as conn:
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                for (table,) in tables:
                    df = normalized(pd.read_sql_query(f'SELECT * FROM "{table}"', conn))
                    if score(df) >= 20:
                        candidates.append((score(df), "sqlite", path, table, df))
        except Exception as exc:
            print(f"[WARN] {path}: {exc}")
    for path in base.rglob("*.csv"):
        if path.name.startswith("FINAL_"):
            continue
        try:
            df = normalized(pd.read_csv(path))
            if score(df) >= 20:
                candidates.append((score(df), "csv", path, "", df))
        except Exception as exc:
            print(f"[WARN] {path}: {exc}")

if not candidates:
    raise SystemExit("No completed multi-prototype result table/CSV with round and ASR was found.")
_, kind, source, table, df = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
df["round"] = pd.to_numeric(df["round"], errors="coerce")
df = df.dropna(subset=["round"]); df["round"] = df["round"].astype(int)
for col in ("clean_acc", "asr"):
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce")
preferred = ["round", "clean_acc", "asr", "selected_clients", "selected_malicious_clients",
             "attack_active", "average_train_loss", "psnr", "ssim", "linf"]
df = df[[c for c in preferred if c in df] + [c for c in df if c not in preferred]].sort_values("round")
all_path = SAVE_DIR / "FINAL_all_round_metrics.csv"
key_path = SAVE_DIR / "FINAL_key_rounds.csv"
readme_path = SAVE_DIR / "README_where_to_open.txt"
df.to_csv(all_path, index=False, encoding="utf-8-sig")
df[df["round"].isin(KEY_ROUNDS)].to_csv(key_path, index=False, encoding="utf-8-sig")
readme_path.write_text(
    "打开此关键轮次文件：\n" + str(key_path) + "\n\n完整每轮结果：\n" + str(all_path)
    + f"\n\n自动来源：{kind} {source}" + (f" table={table}" if table else "") + "\n",
    encoding="utf-8",
)
print(f"Exported {key_path}\nExported {all_path}\nSource: {kind} {source} {table}")
