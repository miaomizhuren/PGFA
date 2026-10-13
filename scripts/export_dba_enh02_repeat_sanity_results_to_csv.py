"""Export the completed DBA enh02 repeat and compare it with the original run."""
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "outputs/sanity_check_dba_enhanced/dba_enh02_size6_pr08_w2_s60_e80_r100/metrics_dba_enh02.db"
REPEAT = ROOT / "outputs/sanity_check_dba_enhanced_repeat/dba_enh02_repeat_size6_pr08_w2_s60_e80_r100/metrics_dba_enh02_repeat.db"
SAVE = ROOT / "outputs/saved_results/sanity_dba_enh02_repeat_20260909"
KEY = [59, 60, 61, 70, 80, 81, 90, 100]


def load(source: Path) -> pd.DataFrame:
    if not source.is_file():
        raise SystemExit(f"Missing completed metrics database: {source}")
    with sqlite3.connect(source) as connection:
        frame = pd.read_sql_query("SELECT * FROM round_metrics ORDER BY round_index", connection)
    frame = frame.rename(columns={"round_index": "round", "clean_accuracy": "clean_acc", "attack_success_rate": "asr"})
    frame["source_file"] = str(source)
    missing = sorted(set(KEY) - set(frame["round"]))
    if missing:
        raise SystemExit(f"Missing key rounds {missing} in {source}")
    return frame


def at(frame: pd.DataFrame, round_index: int, column: str):
    return frame.loc[frame["round"] == round_index, column].iloc[-1]


def comparison_row(name: str, frame: pd.DataFrame, source: Path) -> dict:
    asr80, asr81, asr90 = (at(frame, round_index, "asr") for round_index in (80, 81, 90))
    return {
        "experiment": name,
        "CleanACC@59": at(frame, 59, "clean_acc"),
        **{f"ASR@{round_index}": at(frame, round_index, "asr") for round_index in (60, 61, 70, 80, 81, 90, 100)},
        "ASR_drop_80_to_81": asr80 - asr81,
        "ASR_recovery_81_to_90": asr90 - asr81,
        "source_file": str(source),
    }


def main():
    original, repeat = load(ORIGINAL), load(REPEAT)
    SAVE.mkdir(parents=True, exist_ok=True)
    preferred = ["round", "clean_acc", "asr", "attack_active", "selected_malicious_clients",
                 "num_selected_malicious", "selected_clients", "num_attack_samples",
                 "num_successful_attacks", "protocol_warning", "average_train_loss", "test_loss",
                 "psnr", "ssim", "linf", "recorded_at", "source_file"]
    repeat = repeat[[name for name in preferred if name in repeat] + [name for name in repeat if name not in preferred]]
    repeat.to_csv(SAVE / "FINAL_all_round_metrics.csv", index=False, encoding="utf-8-sig")
    repeat[repeat["round"].isin(KEY)].to_csv(SAVE / "FINAL_key_rounds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        comparison_row("original_enh02_size6_pr08_w2", original, ORIGINAL),
        comparison_row("repeat_enh02_size6_pr08_w2", repeat, REPEAT),
    ]).to_csv(SAVE / "COMPARE_with_original_enh02.csv", index=False, encoding="utf-8-sig")
    (SAVE / "README_where_to_open.txt").write_text(
        "After the repeat finishes, open:\n"
        f"- {SAVE / 'COMPARE_with_original_enh02.csv'}\n"
        f"- {SAVE / 'FINAL_key_rounds.csv'}\n",
        encoding="utf-8",
    )
    print(f"Exported: {SAVE / 'COMPARE_with_original_enh02.csv'}")


if __name__ == "__main__":
    main()
