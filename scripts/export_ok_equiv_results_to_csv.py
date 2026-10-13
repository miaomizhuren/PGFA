from pathlib import Path
import sqlite3
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "outputs/saved_results/ok_equiv_reproduce_20260908"
EXTRA_DIR = ROOT / "outputs/reproduce_best/reproduce_pbu_ok_equiv_s60_e110_seed2026"

KEY_ROUNDS = [110, 111, 120, 130, 150, 180, 200, 240, 250]

FINAL_KEY = SAVE_DIR / "FINAL_key_rounds.csv"
FINAL_ALL = SAVE_DIR / "FINAL_all_round_metrics.csv"
README = SAVE_DIR / "README_where_to_open.txt"

SAVE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_col(c: str) -> str:
    x = c.strip()
    low = x.lower()
    low = low.replace("@", "_at_")
    low = re.sub(r"[^a-z0-9]+", "_", low).strip("_")
    aliases = {
        "round_index": "round",
        "round_id": "round",
        "global_round": "round",
        "clean_accuracy": "clean_acc",
        "cleanacc": "clean_acc",
        "clean_acc_percent": "clean_acc",
        "accuracy": "clean_acc",
        "acc": "clean_acc",
        "attack_success_rate": "asr",
        "attack_success": "asr",
        "backdoor_asr": "asr",
        "asr_percent": "asr",
    }
    return aliases.get(low, low)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col(str(c)) for c in df.columns]

    for c in list(df.columns):
        lc = c.lower()
        if c != "round" and ("round" == lc or lc.endswith("_round") or "global_round" in lc):
            df = df.rename(columns={c: "round"})
        if c != "clean_acc" and ("clean" in lc and ("acc" in lc or "accuracy" in lc)):
            df = df.rename(columns={c: "clean_acc"})
        if c != "asr" and ("asr" in lc or "attack_success" in lc):
            df = df.rename(columns={c: "asr"})
    return df


def score_df(df: pd.DataFrame) -> int:
    cols = set(df.columns)
    score = 0
    if "round" in cols:
        score += 10
    if "asr" in cols:
        score += 10
    if "clean_acc" in cols:
        score += 8
    if "method" in cols:
        score += 2
    if "seed" in cols:
        score += 2
    if "fixed_margin" in cols:
        score += 2
    if "dynamic_margin" in cols:
        score += 2
    return score


def valid_result_df(df: pd.DataFrame) -> bool:
    return "round" in set(df.columns) and "asr" in set(df.columns)


candidates = []

for base in [SAVE_DIR, EXTRA_DIR]:
    if not base.exists():
        continue
    for db_path in list(base.rglob("*.db")) + list(base.rglob("*.sqlite")) + list(base.rglob("*.sqlite3")):
        try:
            with sqlite3.connect(db_path) as conn:
                tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table';", conn
                )["name"].tolist()
                for table in tables:
                    try:
                        df = normalize_df(pd.read_sql_query(f'SELECT * FROM "{table}"', conn))
                        if valid_result_df(df):
                            candidates.append({
                                "source_type": "sqlite", "source": str(db_path),
                                "table": table, "score": score_df(df), "df": df,
                            })
                    except Exception as exc:
                        print(f"[WARN] failed reading table {table} from {db_path}: {exc}")
        except Exception as exc:
            print(f"[WARN] failed opening db {db_path}: {exc}")

for base in [SAVE_DIR, EXTRA_DIR]:
    if not base.exists():
        continue
    for csv_path in base.rglob("*.csv"):
        if csv_path.name.startswith("FINAL_"):
            continue
        try:
            df = normalize_df(pd.read_csv(csv_path))
            if valid_result_df(df):
                candidates.append({
                    "source_type": "csv", "source": str(csv_path),
                    "table": "", "score": score_df(df), "df": df,
                })
        except Exception as exc:
            print(f"[WARN] failed reading csv {csv_path}: {exc}")

if not candidates:
    README.write_text(
        "没有自动找到包含 round 和 asr 的数据库表或 CSV。\n"
        f"已扫描目录：\n{SAVE_DIR}\n{EXTRA_DIR}\n", encoding="utf-8"
    )
    raise SystemExit("ERROR: no result table/csv found with round + asr")

candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
best = candidates[0]
df = best["df"].copy()
df["round"] = pd.to_numeric(df["round"], errors="coerce")
df = df.dropna(subset=["round"])
df["round"] = df["round"].astype(int)
df = df.sort_values("round")

for column in ["clean_acc", "asr", "fixed_margin", "dynamic_margin"]:
    if column in df.columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            df[column] = numeric

preferred = [
    "round", "method", "seed", "target_label", "clean_acc", "asr",
    "fixed_target_sim", "fixed_source_sim", "fixed_margin",
    "dynamic_target_sim", "dynamic_source_sim", "dynamic_margin",
    "prototype_version", "feature_layer", "num_eval_samples",
    "num_source_samples", "num_target_excluded",
]
ordered_cols = [column for column in preferred if column in df.columns]
ordered_cols += [column for column in df.columns if column not in preferred]
df = df[ordered_cols]

df.to_csv(FINAL_ALL, index=False, encoding="utf-8-sig")
key_df = df[df["round"].isin(KEY_ROUNDS)].copy()
key_df.to_csv(FINAL_KEY, index=False, encoding="utf-8-sig")

readme = [
    "OK 等价复现实验结果已经整理完成。", "", "用户最应该打开这个文件：",
    str(FINAL_KEY), "", "完整每轮结果文件：", str(FINAL_ALL), "",
    "自动选择的数据来源：", f"source_type: {best['source_type']}",
    f"source: {best['source']}",
]
if best["table"]:
    readme.append(f"table: {best['table']}")
readme += [f"score: {best['score']}", "", "关键轮次：", ", ".join(map(str, KEY_ROUNDS)), "", "候选结果来源列表："]
for index, item in enumerate(candidates[:20], 1):
    readme.append(
        f"{index}. score={item['score']} type={item['source_type']} "
        f"source={item['source']} table={item['table']}"
    )
README.write_text("\n".join(readme), encoding="utf-8")

print("==== EXPORT DONE ====")
print(f"Best source type: {best['source_type']}")
print(f"Best source: {best['source']}")
if best["table"]:
    print(f"Best table: {best['table']}")
print(f"Saved key rounds: {FINAL_KEY}")
print(f"Saved all rounds: {FINAL_ALL}")
print(f"Saved README: {README}")
print("\n==== KEY ROUNDS PREVIEW ====")
print(key_df.to_string(index=False))
