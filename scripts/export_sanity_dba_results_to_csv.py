"""Export the completed DBA sanity run without changing its source database."""
from pathlib import Path
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/sanity_check_baselines/sanity_dba_s60_e80_r100/metrics_sanity_dba.db"
DESTINATION = ROOT / "outputs/saved_results/sanity_dba_20260909"
KEY_ROUNDS = [59, 60, 61, 70, 80, 81, 90, 100]


def main():
    if not SOURCE.is_file():
        raise SystemExit(f"DBA sanity metrics database not found: {SOURCE}")

    with sqlite3.connect(SOURCE) as connection:
        frame = pd.read_sql_query("SELECT * FROM round_metrics ORDER BY round_index", connection)

    frame = frame.rename(columns={"round_index": "round", "clean_accuracy": "clean_acc", "attack_success_rate": "asr"})
    frame.insert(len(frame.columns), "source_file", str(SOURCE))
    preferred = [
        "round", "clean_acc", "asr", "attack_active", "selected_malicious_clients",
        "num_selected_malicious", "selected_clients", "num_attack_samples",
        "num_successful_attacks", "protocol_warning", "average_train_loss", "test_loss",
        "psnr", "ssim", "linf", "recorded_at", "source_file",
    ]
    frame = frame[[column for column in preferred if column in frame] +
                  [column for column in frame if column not in preferred]]

    missing = sorted(set(KEY_ROUNDS) - set(frame["round"]))
    if missing:
        raise SystemExit(f"Missing required key rounds: {missing}")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    all_metrics = DESTINATION / "FINAL_all_round_metrics.csv"
    key_metrics = DESTINATION / "FINAL_key_rounds.csv"
    frame.to_csv(all_metrics, index=False, encoding="utf-8-sig")
    frame[frame["round"].isin(KEY_ROUNDS)].to_csv(key_metrics, index=False, encoding="utf-8-sig")
    (DESTINATION / "README_where_to_open.txt").write_text(
        "DBA sanity exported results:\n"
        f"- Key rounds: {key_metrics}\n"
        f"- All rounds: {all_metrics}\n"
        f"- Immutable source database: {SOURCE}\n",
        encoding="utf-8",
    )
    print(f"Exported: {key_metrics}")
    print(f"Exported: {all_metrics}")


if __name__ == "__main__":
    main()
