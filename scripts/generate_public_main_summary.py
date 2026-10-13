#!/usr/bin/env python3
"""Rebuild the formal CIFAR-10 main-result table from public roundwise CSVs."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    ("ce_only_uap", "CE-only UAP"),
    ("dba", "DBA"),
    ("neurotoxin", "Neurotoxin-UAP"),
    ("pgfa", "PGFA"),
    ("a3fl_relaxed_strong", "A3FL"),
]
FIELDS = [
    "Dataset", "Method", "Attack_start", "Attack_end", "Observe_start", "Observe_end",
    "PA_AUC_1_500", "ASR_attack_end", "ASR_post_500", "N80", "N90",
    "Roundwise_metrics_source", "Source_SQLite", "Derivation",
]

def main() -> None:
    rows_out = []
    for directory, display_name in SPECS:
        relative = Path("results/main") / directory / "roundwise_metrics.csv"
        with (ROOT / relative).open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        by_round = {int(row["round_index"]): row for row in rows}
        attack_rounds = [round_id for round_id, row in by_round.items() if row["attack_active"] == "1"]
        attack_start, attack_end = min(attack_rounds), max(attack_rounds)
        post_rounds = list(range(attack_end + 1, attack_end + 501))
        if any(round_id not in by_round for round_id in post_rounds):
            raise RuntimeError(f"{relative}: missing a required post-attack round")
        post_asr = [float(by_round[round_id]["attack_success_rate"]) for round_id in post_rounds]
        source_sqlite = sorted({row["source_sqlite"] for row in rows})
        rows_out.append({
            "Dataset": "CIFAR-10",
            "Method": display_name,
            "Attack_start": attack_start,
            "Attack_end": attack_end,
            "Observe_start": attack_end + 1,
            "Observe_end": post_rounds[-1],
            "PA_AUC_1_500": f"{sum(post_asr) / len(post_asr):.12f}",
            "ASR_attack_end": f"{float(by_round[attack_end]['attack_success_rate']):.12f}",
            "ASR_post_500": f"{post_asr[-1]:.12f}",
            "N80": sum(value >= 80.0 for value in post_asr),
            "N90": sum(value >= 90.0 for value in post_asr),
            "Roundwise_metrics_source": relative.as_posix(),
            "Source_SQLite": "; ".join(source_sqlite),
            "Derivation": "PA-AUC=mean(ASR R3201-R3700); N80/N90=count(ASR>=80/90)",
        })
    destination = ROOT / "results/paper_summaries/01_main_results_cifar10.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"wrote {destination} from {len(rows_out)} public roundwise CSVs")

if __name__ == "__main__":
    main()
