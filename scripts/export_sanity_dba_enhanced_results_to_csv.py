"""Export completed enhanced DBA sanity databases into a comparable CSV table."""
from pathlib import Path
import sqlite3

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/sanity_check_dba_enhanced"
SAVE = ROOT / "outputs/saved_results/sanity_dba_enhanced_20260909"
NAMES = [
    "dba_enh01_size5_pr08_w2_s60_e80_r100", "dba_enh02_size6_pr08_w2_s60_e80_r100",
    "dba_enh03_size5_pr10_w2_s60_e80_r100", "dba_enh04_size6_pr10_w3_s60_e80_r100",
]
KEY = [59, 60, 61, 70, 80, 81, 90, 100]


def value(frame, round_index, column):
    rows = frame.loc[frame["round"] == round_index, column]
    return rows.iloc[-1] if not rows.empty else float("nan")


def level(asr):
    if asr >= 80:
        return "strong"
    if asr >= 60:
        return "usable"
    if asr >= 30:
        return "weak_but_injected"
    return "failed"


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in NAMES:
        config_path = CONFIG_DIR / f"{name}.yaml"
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        source = ROOT / config["experiment"]["output_dir"].replace("./", "") / config["tracking"]["database_name"]
        if not source.is_file():
            raise SystemExit(f"Missing completed run for {name}: {source}")
        with sqlite3.connect(source) as connection:
            frame = pd.read_sql_query("SELECT * FROM round_metrics ORDER BY round_index", connection)
        frame = frame.rename(columns={"round_index": "round", "clean_accuracy": "clean_acc", "attack_success_rate": "asr"})
        frame["source_file"] = str(source)
        if sorted(set(KEY) - set(frame["round"])):
            raise SystemExit(f"Required key round missing for {name}")
        frame[frame["round"].isin(KEY)].to_csv(SAVE / f"FINAL_key_rounds_{name}.csv", index=False, encoding="utf-8-sig")
        dba = config["attack"]["dba"]
        asr80, asr81 = value(frame, 80, "asr"), value(frame, 81, "asr")
        rows.append({
            "experiment": name, "subtrigger_size": dba["subtrigger_size"],
            "poison_ratio": config["attack"]["poison_ratio"], "poison_loss_weight": dba["poison_loss_weight"],
            "attack_start": 60, "attack_end": 80, "total_rounds": 100,
            "CleanACC@59": value(frame, 59, "clean_acc"), "ASR@60": value(frame, 60, "asr"),
            "ASR@61": value(frame, 61, "asr"), "ASR@70": value(frame, 70, "asr"),
            "ASR@80": asr80, "ASR@81": asr81, "ASR@90": value(frame, 90, "asr"),
            "ASR@100": value(frame, 100, "asr"), "ASR_gain_60_to_80": asr80 - value(frame, 60, "asr"),
            "ASR_drop_80_to_81": asr80 - asr81, "DBA_success_level": level(asr80), "source_file": str(source),
        })
    compare = SAVE / "FINAL_compare_4groups.csv"
    pd.DataFrame(rows).to_csv(compare, index=False, encoding="utf-8-sig")
    (SAVE / "README_where_to_open.txt").write_text(
        f"Open this comparison after all four runs complete:\n{compare}\n", encoding="utf-8")
    print(f"Exported: {compare}")


if __name__ == "__main__":
    main()
