#!/usr/bin/env python3
"""Summarize the four α=0.8 post-attack aggregation diagnosis branches.

Run only after all requested branches have completed.  The source run supplies
the shared round-3200 baseline; each branch supplies rounds 3201--3400.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DB = ROOT / ("outputs/cifar10_noniid_pbu_force_fromscratch_3_rerun1/"
                    "alpha0.8_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700/"
                    "metrics_cifar10_noniid_alpha0.8_pbu_fromscratch_clean3000_attack3001_3200_observe3201_3700_rerun1.db")
OUTPUT_ROOT = ROOT / "outputs/noniid_alpha0.8_post_diagnosis_3200_to3400"
BRANCHES = {
    "A_original": "metrics_alpha0.8_postdiag_A_original.db",
    "B_bn_buffer_stable": "metrics_alpha0.8_postdiag_B_bn_buffer_stable.db",
    "C_equal_client_weight": "metrics_alpha0.8_postdiag_C_equal_client_weight.db",
    "D_both": "metrics_alpha0.8_postdiag_D_both.db",
}


def query(db: Path, rounds: range | tuple[int, ...]) -> list[dict]:
    if not db.is_file():
        raise FileNotFoundError(f"missing metrics DB: {db}")
    selected = tuple(rounds)
    marks = ",".join("?" for _ in selected)
    with sqlite3.connect(db) as connection:
        rows = connection.execute(
            "SELECT round_index, clean_accuracy, attack_success_rate, selected_clients, "
            "selected_malicious_clients, attack_active, aggregation_weighting, bn_buffer_policy "
            f"FROM round_metrics WHERE round_index IN ({marks}) ORDER BY round_index", selected,
        ).fetchall()
    return [dict(zip(("round", "clean_acc", "asr", "selected_clients", "selected_malicious", "attack_active", "aggregation_weighting", "bn_buffer_policy"), row)) for row in rows]


def main() -> None:
    baseline = query(SOURCE_DB, (3200,))
    if len(baseline) != 1:
        raise RuntimeError("source metrics must contain exactly round 3200")
    output_rows: list[dict] = []
    for branch, db_name in BRANCHES.items():
        records = query(OUTPUT_ROOT / branch / db_name, range(3201, 3401))
        if len(records) != 200:
            raise RuntimeError(f"{branch}: expected 200 metrics for rounds 3201--3400, found {len(records)}")
        clean = np.asarray([row["clean_acc"] for row in records], dtype=float)
        asr = np.asarray([row["asr"] for row in records], dtype=float)
        row3300 = next(row for row in records if row["round"] == 3300)
        row3400 = next(row for row in records if row["round"] == 3400)
        max_recovery = max(clean[index] - clean[:index + 1].min() for index in range(len(clean)))
        output_rows.append({
            "branch": branch,
            "aggregation_weighting": records[0]["aggregation_weighting"],
            "bn_buffer_policy": records[0]["bn_buffer_policy"],
            "clean_min_3201_3400": clean.min(), "clean_mean_3201_3400": clean.mean(), "clean_std_3201_3400": clean.std(),
            "worst_clean_drop_from_3200": baseline[0]["clean_acc"] - clean.min(), "max_clean_recovery": max_recovery,
            "asr_3200": baseline[0]["asr"], "asr_3300": row3300["asr"], "asr_3400": row3400["asr"],
            "asr_mean_3201_3400": asr.mean(), "asr_min_3201_3400": asr.min(), "asr_max_3201_3400": asr.max(),
        })
    target = OUTPUT_ROOT / "post_diagnosis_summary.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
