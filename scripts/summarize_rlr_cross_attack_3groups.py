#!/usr/bin/env python3
"""Create a read-only combined summary for the four RLR cross-attack runs."""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/rlr_cross_attack_3groups_summary"
POST_START, POST_END = 3201, 3700
KEY_ROUNDS = [3001, 3100, 3200, 3201, 3300, 3400, 3500, 3600, 3700]

SOURCES = [
    {
        "group": "D1",
        "attack": "CE-UAP",
        "config": "configs/rlr_cross_attack_3groups/d1_ce_uap_rlr_clean3000_attack3001_3200_observe3201_3700.yaml",
        "output_dir": "outputs/rlr_cross_attack_3groups/d1_ce_uap_rlr_clean3000_attack3001_3200_observe3201_3700",
    },
    {
        "group": "D2",
        "attack": "Neurotoxin-UAP",
        "config": "configs/rlr_cross_attack_3groups/d2_neurotoxin_uap_rlr_clean3000_attack3001_3200_observe3201_3700.yaml",
        "output_dir": "outputs/rlr_cross_attack_3groups/d2_neurotoxin_uap_rlr_clean3000_attack3001_3200_observe3201_3700",
    },
    {
        "group": "D3",
        "attack": "A3FL-v2 UAP",
        "config": "configs/rlr_cross_attack_3groups/d3_a3fl_v2_uap_rlr_clean3000_attack3001_3200_observe3201_3700.yaml",
        "output_dir": "outputs/rlr_cross_attack_3groups/d3_a3fl_v2_uap_rlr_clean3000_attack3001_3200_observe3201_3700",
    },
    {
        "group": "D4",
        "attack": "DBA-3",
        "config": "configs/rlr_cross_attack_d4_dba3_only/d4_dba3_rlr_clean3000_attack3001_3200_observe3201_3700.yaml",
        "output_dir": "outputs/rlr_cross_attack_3groups/d4_dba3_rlr_clean3000_attack3001_3200_observe3201_3700",
    },
]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(fields: list[str], rows: list[dict]) -> str:
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = ["| " + " | ".join(str(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def longest_run(values: list[float], threshold: float) -> int:
    longest = current = 0
    for value in values:
        if value >= threshold:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def trapz(values: list[float]) -> float:
    return sum((left + right) / 2 for left, right in zip(values, values[1:]))


def load_rows(source: dict) -> tuple[list[dict], Path]:
    output_dir = ROOT / source["output_dir"]
    databases = sorted(output_dir.glob("*.db"))
    if len(databases) != 1:
        raise RuntimeError(f"Expected one database in {output_dir}, found {databases}")
    with sqlite3.connect(databases[0]) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute(
            "SELECT round_index, clean_accuracy, attack_success_rate, attack_active "
            "FROM round_metrics ORDER BY round_index"
        )]
    return rows, databases[0]


def pct(value: float) -> str:
    return f"{value:.2f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries, sources_check, loaded = [], [], []

    for source in SOURCES:
        rows, database = load_rows(source)
        by_round = {row["round_index"]: row for row in rows}
        missing = [round_index for round_index in KEY_ROUNDS if round_index not in by_round]
        post_rows = [by_round[round_index] for round_index in range(POST_START, POST_END + 1)
                     if round_index in by_round]
        post_asr = [row["attack_success_rate"] for row in post_rows]
        if missing or len(post_rows) != POST_END - POST_START + 1:
            raise RuntimeError(f"{source['group']} is incomplete: missing={missing}, post_points={len(post_rows)}")
        if any(value is None or not math.isfinite(value) for row in rows
               for value in (row["clean_accuracy"], row["attack_success_rate"])):
            raise RuntimeError(f"{source['group']} contains missing or non-finite metrics")

        final = by_round[POST_END]
        summaries.append({
            "group": source["group"], "attack": source["attack"], "defense": "RLR",
            "asr_at_attack_end_3200": round(by_round[3200]["attack_success_rate"], 6),
            "asr_at_post_start_3201": round(by_round[3201]["attack_success_rate"], 6),
            "asr_at_final_3700": round(final["attack_success_rate"], 6),
            "post_attack_asr_auc_3201_3700": round(trapz(post_asr), 6),
            "post_attack_asr_mean_3201_3700": round(sum(post_asr) / len(post_asr), 6),
            "post_attack_asr_min_3201_3700": round(min(post_asr), 6),
            "post_attack_asr_max_3201_3700": round(max(post_asr), 6),
            "post_longest_contiguous_asr_ge_80": longest_run(post_asr, 80.0),
            "post_longest_contiguous_asr_ge_50": longest_run(post_asr, 50.0),
            "clean_acc_at_final_3700": round(final["clean_accuracy"], 6),
        })
        sources_check.append({
            "group": source["group"], "attack": source["attack"], "config": source["config"],
            "output_dir": source["output_dir"], "sqlite_db": str(database.relative_to(ROOT)),
            "min_round": rows[0]["round_index"], "max_round": rows[-1]["round_index"],
            "round_metric_records": len(rows), "post_attack_records_3201_3700": len(post_rows),
            "key_rounds_present": "yes", "numeric_quality_check": "pass",
        })
        loaded.append((source, by_round))

    summary_fields = list(summaries[0])
    write_csv(OUT / "rlr_cross_attack_summary.csv", summary_fields, summaries)
    write_csv(OUT / "rlr_cross_attack_source_check.csv", list(sources_check[0]), sources_check)

    key_rows = []
    for round_index in KEY_ROUNDS:
        phase = {
            3001: "attack_start", 3100: "attack_mid", 3200: "attack_end", 3201: "post_start",
            3300: "post_100", 3400: "post_200", 3500: "post_300", 3600: "post_400", 3700: "post_500_final",
        }[round_index]
        row = {"global_round": round_index, "phase": phase}
        for source, by_round in loaded:
            metric = by_round[round_index]
            row[f"{source['group']} {source['attack']}_clean_acc"] = round(metric["clean_accuracy"], 6)
            row[f"{source['group']} {source['attack']}_asr"] = round(metric["attack_success_rate"], 6)
        key_rows.append(row)
    write_csv(OUT / "rlr_cross_attack_key_rounds.csv", list(key_rows[0]), key_rows)

    curves = []
    for relative_round, global_round in enumerate(range(POST_START, POST_END + 1), start=1):
        row = {"relative_post_round": relative_round, "global_round": global_round}
        for source, by_round in loaded:
            metric = by_round[global_round]
            row[f"{source['group']} {source['attack']}_clean_acc"] = round(metric["clean_accuracy"], 6)
            row[f"{source['group']} {source['attack']}_asr"] = round(metric["attack_success_rate"], 6)
        curves.append(row)
    write_csv(OUT / "rlr_cross_attack_post_attack_curves.csv", list(curves[0]), curves)

    table_rows = []
    for row in summaries:
        table_rows.append({
            "Group": row["group"], "Attack": row["attack"], "ASR@3200": pct(row["asr_at_attack_end_3200"]),
            "ASR@3201": pct(row["asr_at_post_start_3201"]), "ASR@3700": pct(row["asr_at_final_3700"]),
            "Post AUC": pct(row["post_attack_asr_auc_3201_3700"]), "Post mean": pct(row["post_attack_asr_mean_3201_3700"]),
            "≥80 contiguous": row["post_longest_contiguous_asr_ge_80"], "≥50 contiguous": row["post_longest_contiguous_asr_ge_50"],
            "Clean@3700": pct(row["clean_acc_at_final_3700"]),
        })
    key_table_rows = []
    for row in key_rows:
        converted = {"Round": row["global_round"], "Phase": row["phase"]}
        for source, _ in loaded:
            name = f"{source['group']} {source['attack']}"
            converted[name] = f"{row[name + '_clean_acc']:.2f}/{row[name + '_asr']:.2f}"
        key_table_rows.append(converted)

    ranked = sorted(summaries, key=lambda row: row["post_attack_asr_auc_3201_3700"])
    report = f"""# RLR Cross-Attack Summary (D1–D4)

## 1. Goal

This read-only summary puts the four completed attack families under the same RLR defense protocol into one table. Clean ACC and ASR are percentages.

## 2. Shared Protocol

All runs use CIFAR-10, ResNet18, IID 100-client federation, 10 clients per round, batch size 64, 3 local epochs, malicious clients `[0, 1, 2]`, attack rounds 3001–3200, and post-attack observation rounds 3201–3700. RLR is active at server aggregation from rounds 3001–3700 with `threshold=4`, `server_lr=1.0`, and `tie_policy=negative`.

## 3. Source Experiments

{markdown_table(["group", "attack", "config", "output_dir", "sqlite_db", "max_round"], [{"group": row["group"], "attack": row["attack"], "config": row["config"], "output_dir": row["output_dir"], "sqlite_db": row["sqlite_db"], "max_round": row["max_round"]} for row in sources_check])}

## 4. Main Summary

`Post AUC` is trapezoidal ASR AUC across the 500 post-attack points (3201–3700), in percentage-round units. Cells with a threshold count report the longest contiguous post-attack duration meeting that ASR threshold.

{markdown_table(list(table_rows[0]), table_rows)}

## 5. Key Rounds

Cells are `Clean ACC / ASR` (%).

{markdown_table(list(key_table_rows[0]), key_table_rows)}

## 6. Result Reading

- By post-attack ASR AUC, the ordering from lowest to highest is **{ranked[0]['group']} {ranked[0]['attack']}** ({pct(ranked[0]['post_attack_asr_auc_3201_3700'])}), **{ranked[1]['group']} {ranked[1]['attack']}** ({pct(ranked[1]['post_attack_asr_auc_3201_3700'])}), **{ranked[2]['group']} {ranked[2]['attack']}** ({pct(ranked[2]['post_attack_asr_auc_3201_3700'])}), and **{ranked[3]['group']} {ranked[3]['attack']}** ({pct(ranked[3]['post_attack_asr_auc_3201_3700'])}).
- D4 DBA-3 remains near 1% ASR throughout the post-attack period under this run's full-trigger DBA evaluation, whereas D1 CE-UAP and D3 A3FL-v2 retain substantially higher post-attack ASR.
- D2 Neurotoxin-UAP has low ASR but also ends with the lowest clean accuracy among the four runs ({pct(next(row['clean_acc_at_final_3700'] for row in summaries if row['group'] == 'D2'))}%).
- These are one controlled run per attack family under a shared RLR setting; they are comparative empirical results, not multi-seed estimates.

## 7. Data Quality Check

All four selected SQLite sources span rounds 3001–3700, contain all requested key rounds, and have 500 post-attack records. No missing or non-finite selected Clean ACC/ASR values were found. Raw databases, checkpoints, configurations, and logs were not modified.

## 8. Files Generated

- `outputs/rlr_cross_attack_3groups_summary/rlr_cross_attack_summary.csv`
- `outputs/rlr_cross_attack_3groups_summary/rlr_cross_attack_key_rounds.csv`
- `outputs/rlr_cross_attack_3groups_summary/rlr_cross_attack_post_attack_curves.csv`
- `outputs/rlr_cross_attack_3groups_summary/rlr_cross_attack_source_check.csv`
- `outputs/rlr_cross_attack_3groups_summary/rlr_cross_attack_report.md`
"""
    (OUT / "rlr_cross_attack_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote summary to {OUT}")


if __name__ == "__main__":
    main()
