"""Export completed Neurotoxin-UAP update-delta mask sanity runs."""
from pathlib import Path
import sqlite3

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/sanity_check_neurotoxin_uap_update_mask"
SAVE_DIR = ROOT / "outputs/saved_results/sanity_neurotoxin_uap_update_mask_20260909"
NAMES = [
    "neuro_update_mask_mr005_s60_e80_r100",
    "neuro_update_mask_mr010_s60_e80_r100",
    "neuro_update_mask_mr020_s60_e80_r100",
]
KEY_ROUNDS = [59, 60, 61, 70, 80, 81, 90, 100]


def value_at(frame, round_index, column):
    values = frame.loc[frame["round"] == round_index, column]
    return None if values.empty else values.iloc[-1]


def difference(left, right):
    return None if left is None or right is None else left - right


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    rows, missing_by_run = [], {}
    for name in NAMES:
        cfg = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
        db = ROOT / cfg["experiment"]["output_dir"].removeprefix("./") / cfg["tracking"]["database_name"]
        if not db.is_file():
            raise SystemExit(f"Missing completed run: {db}")
        with sqlite3.connect(db) as connection:
            frame = pd.read_sql_query("SELECT * FROM round_metrics ORDER BY round_index", connection)
        frame = frame.rename(columns={"round_index": "round", "clean_accuracy": "clean_acc", "attack_success_rate": "asr"})
        frame["round"] = pd.to_numeric(frame["round"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["round"])
        frame["round"] = frame["round"].astype(int)
        missing = sorted(set(KEY_ROUNDS) - set(frame["round"]))
        missing_by_run[name] = missing
        a80, a81, a100 = (value_at(frame, round_index, "asr") for round_index in (80, 81, 100))
        rows.append({
            "experiment": name,
            "mask_ratio": cfg["attack"]["neurotoxin"]["mask_ratio"],
            "mask_apply_mode": cfg["attack"]["neurotoxin"]["mask_apply_mode"],
            "effective_keep_ratio": 1 - cfg["attack"]["neurotoxin"]["mask_ratio"],
            "selected_direction": "high_gradient masked; low_gradient kept",
            **{f"ASR@{round_index}": value_at(frame, round_index, "asr") for round_index in (60, 61, 70, 80, 81, 90, 100)},
            "ASR_drop_80_to_81": difference(a80, a81),
            "ASR_retention_100_vs_80": difference(a100, a80),
            "source_file": str(db),
        })
    output = SAVE_DIR / "FINAL_compare_3groups.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    (SAVE_DIR / "README_where_to_open.txt").write_text(
        f"Open: {output}\n\nMissing key rounds:\n" + "\n".join(
            f"- {name}: " + (", ".join(map(str, missing)) if missing else "none")
            for name, missing in missing_by_run.items()
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Exported: {output}")


if __name__ == "__main__":
    main()
