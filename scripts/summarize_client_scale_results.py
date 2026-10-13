#!/usr/bin/env python3
"""Create a non-destructive comparison of the client-scale experiments.

The script is safe to run before a diagnostic experiment starts: missing or
incomplete metric databases are reported as ``not_completed`` rather than
treated as an error.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/summaries/client_scale"
ROUNDS = tuple(range(100, 401, 10))
EXPERIMENTS = (
    ("20c10 e1 mainexp_04", "configs/mainexp/mainexp_04_proto_uap_layer4_proposed.yaml"),
    ("100c10 e1", "configs/client_scale/client_scale_100c10_proto_uap_layer4_proposed.yaml"),
    ("100c10 e3", "configs/client_scale/client_scale_100c10_e3_proto_uap_layer4_proposed.yaml"),
    ("100c10 e5", "configs/client_scale/client_scale_100c10_e5_proto_uap_layer4_proposed.yaml"),
    ("100c10 e5 r400 late", "configs/client_scale/client_scale_100c10_e5_r400_late_proto_uap_layer4_proposed.yaml"),
    ("100c10 e3 r400 late", "configs/client_scale/client_scale_100c10_e3_r400_late_proto_uap_layer4_proposed.yaml"),
)
FIELDS = [
    "method", "config_path", "status", "num_clients", "clients_per_round",
    "participation_rate", "local_epochs", "attack_start_round", "attack_end_round",
    "total_rounds", "malicious_clients",
    *[f"CleanACC@{round_index}" for round_index in ROUNDS],
    *[f"ASR@{round_index}" for round_index in ROUNDS],
    "PSNR", "SSIM", "L_inf", "AUC@100-150", "AUC@100-200", "AUC@220-400",
    "Half-life", "note",
]


def fmt(value: float | None, decimals: int = 2) -> str:
    return "" if value is None else f"{float(value):.{decimals}f}"


def auc(data: dict[int, dict[str, float | None]], start_round: int, end_round: int) -> str:
    """Return mean trapezoidal ASR from ``start_round`` through ``end_round``."""
    points = [
        (round_index, data[round_index]["asr"])
        for round_index in sorted(data)
        if start_round <= round_index <= end_round and data[round_index]["asr"] is not None
    ]
    if len(points) < 2 or points[0][0] != start_round or points[-1][0] != end_round:
        return ""
    area = sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2
        for left, right in zip(points, points[1:])
    )
    return fmt(area / (end_round - start_round))


def load_metrics(database: Path) -> tuple[dict[int, dict[str, float | None]], str]:
    if not database.exists():
        return {}, f"not completed; missing metrics DB: {database.relative_to(ROOT)}"
    try:
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                "SELECT round_index, clean_accuracy, attack_success_rate, psnr, ssim, linf "
                "FROM round_metrics ORDER BY round_index"
            ).fetchall()
    except sqlite3.Error as error:
        return {}, f"sqlite error: {error}"
    return {
        round_index: {
            "clean": clean, "asr": asr, "psnr": psnr, "ssim": ssim, "linf": linf,
        }
        for round_index, clean, asr, psnr, ssim, linf in rows
    }, ""


def latest_metric(data: dict[int, dict[str, float | None]], key: str) -> float | None:
    return next((data[round_index][key] for round_index in sorted(data, reverse=True)
                 if data[round_index][key] is not None), None)


def make_row(method: str, relative_config: str) -> dict[str, str]:
    config_path = ROOT / relative_config
    empty = {field: "" for field in FIELDS}
    empty.update({"method": method, "config_path": relative_config, "status": "not_completed"})
    if not config_path.exists():
        empty["note"] = f"not completed; missing config: {relative_config}"
        return empty

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    federated, partition, attack = config["federated"], config["partition"], config["attack"]
    database = ROOT / config["experiment"]["output_dir"] / config["tracking"]["database_name"]
    data, note = load_metrics(database)
    last_round = max(data, default=0)
    status = "complete" if last_round >= federated["num_rounds"] else "not_completed"
    if not note:
        note = (f"complete through Round {last_round}" if status == "complete"
                else f"not completed; last recorded round: {last_round}")

    row = {
        **empty,
        "status": status,
        "num_clients": str(partition["num_clients"]),
        "clients_per_round": str(federated["clients_per_round"]),
        "participation_rate": f"{100 * federated['clients_per_round'] / partition['num_clients']:.0f}%",
        "local_epochs": str(federated["local_epochs"]),
        "attack_start_round": str(attack["attack_start_round"]),
        "attack_end_round": str(attack["attack_end_round"]),
        "total_rounds": str(federated["num_rounds"]),
        "malicious_clients": str(attack["malicious_client_ids"]),
        "note": note,
    }
    row.update({f"CleanACC@{round_index}": fmt(data.get(round_index, {}).get("clean"))
                for round_index in ROUNDS})
    row.update({f"ASR@{round_index}": fmt(data.get(round_index, {}).get("asr"))
                for round_index in ROUNDS})
    row.update({
        "PSNR": fmt(latest_metric(data, "psnr")),
        "SSIM": fmt(latest_metric(data, "ssim"), 4),
        "L_inf": fmt(latest_metric(data, "linf"), 4),
        "AUC@100-150": auc(data, 100, 150),
        "AUC@100-200": auc(data, 100, 200),
        "AUC@220-400": auc(data, 220, 400),
        "Half-life": next(
            (str(round_index) for round_index in sorted(data)
             if round_index >= attack["attack_end_round"] and data[round_index]["asr"] is not None
             and data[round_index]["asr"] < 50),
            f">{federated['num_rounds']}" if status == "complete" else "",
        ),
    })
    return row


def value(row: dict[str, str], field: str) -> str:
    return row[field] or "N/A"


def write_markdown(rows: list[dict[str, str]]) -> None:
    lines = [
        "# Client-scale summary", "",
        "| Setting | Status | Attack end | Clean ACC@end | Clean ACC@final | ASR@end | ASR@final | AUC@post-attack | Half-life |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        attack_end = row["attack_end_round"]
        final_round = row["total_rounds"]
        post_attack_auc = "AUC@220-400" if attack_end == "220" else "AUC@100-200"
        lines.append(
            f"| {row['method']} | {row['status']} | {attack_end} | "
            f"{value(row, f'CleanACC@{attack_end}')} | {value(row, f'CleanACC@{final_round}')} | "
            f"{value(row, f'ASR@{attack_end}')} | {value(row, f'ASR@{final_round}')} | "
            f"{value(row, post_attack_auc)} | {value(row, 'Half-life')} |"
        )
    lines.extend(["", "## Comparison analysis", ""])
    for row in rows:
        attack_end = row["attack_end_round"]
        final_round = row["total_rounds"]
        post_attack_auc = "AUC@220-400" if attack_end == "220" else "AUC@100-200"
        lines.append(
            f"- {row['method']} ({row['status']}): CleanACC@attack_end ({attack_end})="
            f"{value(row, f'CleanACC@{attack_end}')}, CleanACC@final ({final_round})="
            f"{value(row, f'CleanACC@{final_round}')}, ASR@attack_end={value(row, f'ASR@{attack_end}')}, "
            f"ASR@final={value(row, f'ASR@{final_round}')}, {post_attack_auc}="
            f"{value(row, post_attack_auc)}, Half-life={value(row, 'Half-life')}."
        )
    pending = [row["method"] for row in rows if row["status"] != "complete"]
    if pending:
        lines.extend(["", f"Pending/not completed: {', '.join(pending)}. Metrics are N/A until runs finish."])
    (OUT / "client_scale_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [make_row(method, config) for method, config in EXPERIMENTS]
    with (OUT / "client_scale_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows)


if __name__ == "__main__":
    main()
