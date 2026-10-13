#!/usr/bin/env python3
"""Export the four-way paired seed-2026 PGFA feature-loss ablation.

This utility is deliberately read-only with respect to experiment inputs.  It
refuses to export unless all four SQLite sources have one finite, unique metric
record for every round 3001--3700 and the two newly run ablations have the
expected attack-state transition.  It does not load models, use CUDA, or alter
checkpoints.
"""

from __future__ import annotations

import csv
import math
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/pgfa_feature_loss_ablation_seed2026"
SOURCES = {
    "CE-only": ROOT / "outputs/main_experiment_clean3000_force/ce_uap_clean3000_attack3001_3200_observe3201_3700/metrics_ce_uap_clean3000_attack3001_3200_observe3201_3700.db",
    "Target-only": OUT / "target_only_seed2026/metrics_pgfa_target_only_seed2026.db",
    "Feature Alignment": OUT / "feature_alignment_seed2026/metrics_pgfa_feature_alignment_seed2026.db",
    "Full PGFA": ROOT / "outputs/pbu_force_clean_stage_sensitivity/clean3000_attack3001_3200_observe3201_3700/metrics_pbu_force_clean3000_attack3001_3200_observe3201_3700.db",
}
NEW_METHODS = {"Target-only", "Feature Alignment"}
EXPECTED = list(range(3001, 3701))


def read_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise SystemExit(f"missing SQLite source: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT round_index, clean_accuracy, attack_success_rate, attack_active, "
            "selected_clients, selected_malicious_clients FROM round_metrics ORDER BY round_index"
        ).fetchall()
    finally:
        conn.close()
    result = [
        {"round": int(r[0]), "clean_accuracy": float(r[1]), "asr": None if r[2] is None else float(r[2]),
         "attack_active": int(r[3]), "selected_clients": r[4], "selected_malicious_clients": r[5]}
        for r in rows
    ]
    rounds = [r["round"] for r in result]
    if rounds != EXPECTED:
        raise SystemExit(f"{path}: expected exactly R3001--R3700 once; got {len(rounds)} records from {min(rounds, default=None)} to {max(rounds, default=None)}")
    for row in result:
        if not math.isfinite(row["clean_accuracy"]) or row["asr"] is None or not math.isfinite(row["asr"]):
            raise SystemExit(f"{path}: non-finite or missing metric at R{row['round']}")
    return result


def metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    by_round = {int(r["round"]): r for r in rows}
    post = [float(by_round[r]["asr"]) for r in range(3201, 3701)]
    post_acc = [float(by_round[r]["clean_accuracy"]) for r in range(3201, 3701)]
    return {
        "ASR@3200": float(by_round[3200]["asr"]),
        "ASR@+100": float(by_round[3300]["asr"]),
        "ASR@+200": float(by_round[3400]["asr"]),
        "ASR@+300": float(by_round[3500]["asr"]),
        "ASR@+400": float(by_round[3600]["asr"]),
        "ASR@+500": float(by_round[3700]["asr"]),
        "PA-AUC": sum(post) / len(post),
        "N80": sum(value >= 80.0 for value in post),
        "N90": sum(value >= 90.0 for value in post),
        "Last50_ASR": sum(post[-50:]) / 50,
        "Post_Clean_ACC": sum(post_acc) / len(post_acc),
    }


def assert_new_attack_states(name: str, rows: list[dict[str, object]]) -> None:
    if name not in NEW_METHODS:
        return
    if any(int(row["attack_active"]) != 1 for row in rows if int(row["round"]) <= 3200):
        raise SystemExit(f"{name}: attack_active is false during the configured attack window")
    if any(int(row["attack_active"]) != 0 for row in rows if int(row["round"]) >= 3201):
        raise SystemExit(f"{name}: attack_active remains true after R3200")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing export: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_rows = {name: read_rows(path) for name, path in SOURCES.items()}
    for name, rows in all_rows.items():
        assert_new_attack_states(name, rows)
    summary = []
    for name in ("CE-only", "Target-only", "Feature Alignment", "Full PGFA"):
        row = {"Method": name, "Seed": 2026, "Source_SQLite": str(SOURCES[name].relative_to(ROOT))}
        row.update(metrics(all_rows[name]))
        summary.append(row)
    comparison_path = OUT / "final_ablation_comparison.csv"
    write_csv(comparison_path, summary, list(summary[0]))

    curve_rows = []
    for name, rows in all_rows.items():
        for row in rows:
            curve_rows.append({
                "Method": name, "Seed": 2026, "round": row["round"],
                "relative_round": int(row["round"]) - 3200,
                "phase": "attack" if int(row["round"]) <= 3200 else "post_attack",
                "Clean_ACC": row["clean_accuracy"], "ASR": row["asr"],
                "attack_active": row["attack_active"],
                "selected_clients": row["selected_clients"],
                "selected_malicious_clients": row["selected_malicious_clients"],
            })
    write_csv(OUT / "four_way_roundwise_curves.csv", curve_rows, list(curve_rows[0]))

    full = next(row for row in summary if row["Method"] == "Full PGFA")
    lines = [
        "# Seed-2026 PGFA feature-loss ablation", "",
        "All metrics are read from the listed SQLite files. PA-AUC is the arithmetic mean of ASR for R3201--R3700 (500 rounds); N80/N90 count those 500 rounds.", "",
        "| Method | Seed | PA-AUC | ASR@+500 | N80 | N90 | Post Clean ACC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['Method']} | 2026 | {row['PA-AUC']:.6f} | {row['ASR@+500']:.6f} | {row['N80']} | {row['N90']} | {row['Post_Clean_ACC']:.6f} |")
    lines.extend(["", "## Full PGFA deltas", ""])
    for row in summary:
        if row["Method"] == "Full PGFA":
            continue
        lines.append(f"- Full PGFA − {row['Method']}: PA-AUC {full['PA-AUC'] - row['PA-AUC']:+.6f}; ASR@+500 {full['ASR@+500'] - row['ASR@+500']:+.6f}.")
    lines.extend(["", "## Validation", "", "- All four inputs contain exactly one finite metric record for each R3001--R3700.", "- Target-only and Feature Alignment were additionally checked for `attack_active=true` in R3001--R3200 and `false` in R3201--R3700.", "- Curves are exported in `four_way_roundwise_curves.csv` without interpolation or cross-seed mixing."])
    report = OUT / "final_ablation_report.md"
    if report.exists():
        raise SystemExit(f"refusing to overwrite existing report: {report}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {comparison_path}")
    print(f"WROTE {OUT / 'four_way_roundwise_curves.csv'}")
    print(f"WROTE {report}")


if __name__ == "__main__":
    main()
